use super::*;
use eredu_core::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};

#[derive(Debug)]
struct Account {
    live: Arc<AtomicBool>,
    used: Arc<AtomicUsize>,
    refuse: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        if self.refuse.load(Ordering::SeqCst) {
            return Err(HostMetadataFundingError::Unavailable);
        }
        self.used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes)
            })
            .map(|_| ())
            .map_err(|_| HostMetadataFundingError::Overflow)
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.live.store(false, Ordering::SeqCst);
    }
}
struct Funded<'a> {
    transport: &'a Transport,
    funding: HostMetadataFunding,
    borrowed: AtomicUsize,
}
impl ConsensusTransport for Funded<'_> {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        self.transport.participant_count()
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Self::Error> {
        panic!("a parameter operation must use its bounded transport")
    }
}
impl BoundedConsensusTransport for Funded<'_> {
    type Completion = Done;
    type GatherOutput = (usize, Vec<u32>);
    fn metadata_funding(&self) -> Option<&HostMetadataFunding> {
        Some(&self.funding)
    }
    fn submit_all_gather_words(
        &self,
        words: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Done>, Self::Error> {
        self.transport.submit_all_gather_words(words)
    }
    fn resolve_all_gather_words(&self, _: Self::GatherOutput) -> Result<Vec<u32>, Self::Error> {
        panic!("funded protocol validation must borrow the completed destination")
    }
    fn with_resolved_all_gather_words<T, E, F>(
        &self,
        output: Self::GatherOutput,
        validate: F,
    ) -> Result<Result<T, E>, Self::Error>
    where
        F: FnOnce(&[u32]) -> Result<T, E>,
    {
        self.borrowed.fetch_add(1, Ordering::SeqCst);
        let words = self.transport.resolve_all_gather_words(output)?;
        Ok(validate(&words))
    }
}
impl ParameterOperationTransport for Funded<'_> {
    fn parameter_rank(&self) -> usize {
        self.transport.parameter_rank()
    }
    fn parameter_setup(&self) -> CommunicationSessionIdentity {
        self.transport.parameter_setup()
    }
    fn parameter_wait(&self) -> Result<BoundedCompletionWait, ParameterError> {
        self.transport.parameter_wait()
    }
    fn estimate_parameter_gather(&self, n: usize) -> Result<CaptureUsage, ParameterError> {
        self.transport.estimate_parameter_gather(n)
    }
    fn ensure_parameter_active(&self) -> Result<(), BackendFailure> {
        self.transport.ensure_parameter_active()
    }
    fn fail_parameter_operation(&self, error: &Error) {
        self.transport.fail_parameter_operation(error);
    }
}

#[test]
fn funded_parameter_frames_borrow_completed_words_and_keep_escaped_failure_payer() {
    ranks(Fault::None, |rank, transport, owner| {
        let live = Arc::new(AtomicBool::new(true));
        let used = Arc::new(AtomicUsize::new(0));
        let funded = Funded {
            transport,
            borrowed: AtomicUsize::new(0),
            funding: HostMetadataFunding::new(Account {
                live: live.clone(),
                used: used.clone(),
                refuse: Arc::new(AtomicBool::new(false)),
            })
            .unwrap(),
        };
        let before = used.load(Ordering::SeqCst);
        let result = owner.read(
            &funded,
            binding(),
            &[19; 32],
            ParameterOperationKind::Query,
            Ok(ParameterReadPreparation::new((), [41; 32], 2)),
            &mut Budget::default(),
            |_| Ok(original(rank).into_iter().map(f32::to_bits).collect()),
            |rows| {
                assert_eq!(rows.len(), 3);
                for (rank, row) in rows.iter().enumerate() {
                    assert_eq!(
                        row.iter().copied().map(f32::from_bits).collect::<Vec<_>>(),
                        original(rank)
                    );
                }
                Ok(())
            },
        );
        result.unwrap();
        assert!(used.load(Ordering::SeqCst) > before);
        assert!(funded.borrowed.load(Ordering::SeqCst) >= 7);
        let used_before_error = used.load(Ordering::SeqCst);
        let error = owner
            .read(
                &funded,
                binding(),
                &[23; 32],
                ParameterOperationKind::Query,
                Err::<ParameterReadPreparation<()>, _>(ParameterError::Overflow),
                &mut Budget::default(),
                |_| unreachable!(),
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(completed_rejection(&error));
        assert!(used.load(Ordering::SeqCst) > used_before_error);
        assert!(!transport.fenced.load(Ordering::SeqCst));
        drop(funded);
        assert!(
            live.load(Ordering::SeqCst),
            "escaped typed failure keeps its closed protocol owner"
        );
        let spent = used.load(Ordering::SeqCst);
        drop(error);
        assert!(!live.load(Ordering::SeqCst));
        assert_eq!(
            used.load(Ordering::SeqCst),
            spent,
            "completed rejection never refunds attempts"
        );
    });
}

#[test]
fn protocol_metadata_refusal_precedes_submission_callbacks_and_attempt_mutation() {
    ranks(Fault::None, |_, transport, owner| {
        let live = Arc::new(AtomicBool::new(true));
        let used = Arc::new(AtomicUsize::new(0));
        let refuse = Arc::new(AtomicBool::new(false));
        let funded = Funded {
            transport,
            borrowed: AtomicUsize::new(0),
            funding: HostMetadataFunding::new(Account {
                live: live.clone(),
                used: used.clone(),
                refuse: refuse.clone(),
            })
            .unwrap(),
        };
        let before = used.load(Ordering::SeqCst);
        let usage = owner.usage().unwrap();
        refuse.store(true, Ordering::SeqCst);
        let error = owner
            .read(
                &funded,
                binding(),
                &[19; 32],
                ParameterOperationKind::Query,
                Ok(ParameterReadPreparation::new((), [41; 32], 2)),
                &mut Budget::default(),
                |_| unreachable!(),
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ParameterError::Metadata(HostMetadataFundingError::Unavailable)
        ));
        assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
        assert_eq!(owner.usage().unwrap(), usage);
        assert_eq!(used.load(Ordering::SeqCst), before);
        assert!(!transport.fenced.load(Ordering::SeqCst));
        drop(funded);
        assert!(
            !live.load(Ordering::SeqCst),
            "fixed metadata refusal owns no unpaid heap payload"
        );
    });
}
