use super::*;

impl OriginalParallelControlProjection {
    // Exercise the real lexical installation and its account-only aliases after
    // an actual admitted role has settled. No request/native authority is
    // fabricated: the intentionally absent request always refuses upgrade.
    pub(crate) fn exercise_inactive_retirement(raw: OriginalTextMetadataCustody) -> Self {
        use eredu_runtime::working_memory::{InferenceExecutionIdentity, MemoryLedger};
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let world = NativeGroup::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let descriptor = CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(7),
            0,
            vec![0],
            Some(0),
            eredu_runtime::CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::barrier(true),
            ])
            .unwrap(),
        )
        .unwrap();
        let manifest = CommunicationManifest::new(1, 0, vec![descriptor], vec![])
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let actual = ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
        let authority = PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
        let pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
        let funding = pool
            .prepare_workspace_metadata(
                &InferenceExecutionIdentity::default(),
                crate::memory_fixture::resolved_limits(1 << 20),
            )
            .unwrap();
        let source = actual
            .bind_original_source(&manifest, &world, &authority, &funding)
            .unwrap();
        funding
            .reserve_metadata(Self::retirement_control_bytes().unwrap())
            .unwrap();
        let custody = Custody {
            source: source.source().clone(),
            raw: raw.into(),
            funding,
            model_funding: None,
        };
        let fallback = eredu_core::SharedBackendFailure::new(
            eredu_core::BackendFailureKind::InvalidSession,
            ControlCause::Closed,
        );
        let activation = Rc::new(Activation {
            active: Cell::new(true),
            custody: custody.clone(),
        });
        let installation = OriginalParallelControlInstallation(Some(activation.clone()));
        let projection = Self {
            model_roots: None,
            owner: Weak::new(),
            active: Some(activation),
            fallback,
            custody,
        };
        assert!(projection.is_active());
        assert!(
            projection.upgrade().is_err(),
            "lexical custody supplies no request authority"
        );
        let escaped = projection.clone();
        let slot = RefCell::new(Some(projection));
        assert!(Self::take_inactive(&mut slot.borrow_mut()).is_none());
        assert!(
            slot.borrow().is_some(),
            "active installation must remain installed"
        );
        drop(installation); // the same close-on-drop path as actual submissions
        assert!(!escaped.is_active());
        assert!(
            escaped.upgrade().is_err(),
            "escaped closed view cannot submit"
        );
        let retired = Self::take_inactive(&mut slot.borrow_mut()).expect("closed slot retires");
        assert!(slot.borrow().is_none());
        assert!(Self::take_inactive(&mut slot.borrow_mut()).is_none());
        drop(retired); // outside the slot loan, retaining the independent escape
        drop(source);
        drop((actual, authority, world, stream));
        assert!(
            pool.fixture_host_charge().unwrap() > 0,
            "escaped view retains its source payer"
        );
        escaped
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
