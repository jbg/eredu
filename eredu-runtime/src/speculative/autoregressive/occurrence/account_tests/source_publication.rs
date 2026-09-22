//! Real selected role/ticket and source-bank lifecycle; the fixed payload is a
//! neutral accounting oracle, not a native source or kernel qualification.
use super::*;
use std::cell::RefCell;

type Registration = NativeStorageRegistration<u32>;
struct Completed {
    attachment: RefCell<Option<Registration>>,
    payload: [u8; 16],
    key: u32,
    fail: bool,
    _receipt: OriginalHostSourceReceipt,
}
struct Producer {
    custody: OriginalHostSourceCustody,
    key: u32,
    fail: bool,
    bytes: u64,
}
impl OriginalHostSourceConstruction for Producer {
    type Key = u32;
    type Completed = Completed;
    type Output = Completed;
    type Attachment = Registration;
    type Error = WorkingMemoryError;
    type Observation<'a> = &'a Completed;
    fn source_custody(&self) -> OriginalHostSourceCustody {
        self.custody.clone()
    }
    fn storage_bytes(&self) -> Result<(u64, u64), WorkingMemoryError> {
        Ok((self.bytes, 16))
    }
    fn create(self, receipt: OriginalHostSourceReceipt) -> Result<Completed, WorkingMemoryError> {
        Ok(Completed {
            attachment: RefCell::new(None),
            payload: [37; 16],
            key: self.key,
            fail: self.fail,
            _receipt: receipt,
        })
    }
    fn observe(value: &Completed) -> Result<&Completed, WorkingMemoryError> {
        Ok(value)
    }
    fn describe(value: &&Completed) -> Option<(u32, u64)> {
        Some((value.key, value.payload.len() as u64))
    }
    fn prepare_attachment(value: Registration) -> Result<Registration, WorkingMemoryError> {
        Ok(value)
    }
    fn registration(value: &Registration) -> &Registration {
        value
    }
    fn attach(
        value: &Completed,
        attachment: Registration,
    ) -> Result<(), (WorkingMemoryError, Registration)> {
        if value.fail {
            return Err((WorkingMemoryError::AlreadyStarted, attachment));
        }
        *value.attachment.borrow_mut() = Some(attachment);
        Ok(())
    }
    fn into_output(value: Completed) -> Completed {
        value
    }
}

#[test]
fn original_ticket_source_publication_and_failed_attachment_retain_exact_charge() {
    let bytes = match OriginalHostSourceBank::publication_control_bytes::<Producer>(0) {
        Ok(bytes) => bytes + std::mem::size_of::<Completed>() as u64,
        Err(WorkingMemoryError::UnknownBound) => {
            assert_ne!(
                std::env::var("EREDU_REQUIRE_IMMUTABLE_SOURCE_QUALIFICATION").as_deref(),
                Ok("1")
            );
            return;
        }
        Err(cause) => panic!("source controls: {cause}"),
    };
    for (fail, funded_table) in [(false, false), (true, false), (false, true), (true, true)] {
        let selected = selected();
        let config = SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 1,
            ..Default::default()
        };
        let schedule = AutoregressiveSchedulePlan::new(
            &selected,
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(16).unwrap(),
            &config,
            SpeculativeSchedulerOptions::default(),
        )
        .unwrap();
        let invocation =
            AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
        let report = report(
            schedule
                .workspace_geometry(2, invocation, NonZeroU64::new(1).unwrap())
                .unwrap(),
        );
        let pool = crate::working_memory::memory_fixture::host_ledger(1 << 24, 0).unwrap();
        let request = OriginalSpeculativeRequest::prepare(
            &pool,
            &InferenceExecutionIdentity::default(),
            &schedule,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 24),
        )
        .unwrap();
        let mut cursor = schedule.into_cursor();
        let source = HostSourceConstructionFacts::new(bytes, 1, 1).unwrap();
        let requirements = if funded_table {
            let construction = pool.prepare_construction_metadata().unwrap();
            let mut table = SpeculativeHostSourceSpans::new(1, construction.funding()).unwrap();
            table.push(Some(source)).unwrap();
            assert_eq!(table.push(None), Err(WorkingMemoryError::IdentityMismatch));
            assert_eq!(table.as_slice(), &[Some(source)]);
            let funding = construction.seal().unwrap();
            let before_move = pool.payload_used_bytes().unwrap();
            let raw = vec![Some(source)];
            let raw_table_bytes = (raw.capacity()
                * std::mem::size_of::<Option<HostSourceConstructionFacts>>())
                as u64;
            let raw_requirements = requirements(report.span_workspace_plan())
                .with_host_source_spans(raw)
                .unwrap();
            let prepared = requirements(report.span_workspace_plan())
                .with_funded_host_source_spans(table)
                .unwrap();
            assert_eq!(
                raw_requirements.allocation_bytes()[3],
                prepared.allocation_bytes()[3] + raw_table_bytes,
                "only the separately prepaid table backing leaves the role allowance"
            );
            drop((raw_requirements, funding));
            assert_eq!(
                pool.payload_used_bytes().unwrap(),
                before_move,
                "moving the table retains its actual construction payer"
            );
            prepared
        } else {
            requirements(report.span_workspace_plan())
                .with_host_source_constructions(source)
                .unwrap()
        };
        let role = request
            .reserve_role(cursor.claim(2, invocation).unwrap(), requirements)
            .unwrap();
        role.claim_neural_bank(0).unwrap();
        let mut bank = role.take_host_source_constructions().unwrap().unwrap();
        let before = pool.payload_used_bytes().unwrap();
        let result = bank.construct(Producer {
            custody: role.budget_custody().into(),
            key: 17,
            fail,
            bytes,
        });
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            before,
            "prepaid publication does not charge source twice"
        );
        assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (0, 0));
        assert!(
            role.take_host_source_constructions().is_err(),
            "publication cannot replenish source issuance"
        );
        request.close().unwrap();
        drop((bank, role, request));
        assert!(
            pool.payload_used_bytes().unwrap() > 0,
            "source/error owns the exact accepted ticket"
        );
        match result {
            Ok(output) => {
                assert!(!fail);
                assert_eq!(output.payload, [37; 16]);
                // Real canonical pin shares the prepaid source origin, even
                // after the request/header have retired. It grants no source birth.
                let alias = pool.pin_registered_storage([(17u32, 16)]).unwrap();
                drop(output);
                assert!(
                    pool.payload_used_bytes().unwrap() > 0,
                    "final canonical alias keeps the ticket"
                );
                drop(alias);
            }
            Err(error) => {
                assert!(fail);
                let (unstarted, retained) = error.into_parts();
                assert!(unstarted.is_none());
                assert!(matches!(
                    retained.cause(),
                    OriginalHostSourceFailureCause::Native(WorkingMemoryError::AlreadyStarted)
                ));
                assert!(retained.completed().is_some());
                drop(retained);
            }
        }
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}
