use super::*;
use crate::backend::runtime::checkpoint::store::{GgufHostCopyCause, PreparedGgufHostCopyFailure};
use eredu_runtime::working_memory::{HostDestinationCause, OriginalHostSourceBank};

impl OriginalGgufMissFixture {
    /// Control-only diagnostic; copied payload is deliberately separate.
    pub(crate) fn source_arena_layouts(&self) -> Vec<u64> {
        let plan = self.source.conversion_plan(&Self::request()).unwrap();
        (0..plan.conversion().outputs().len())
            .map(|ordinal| {
                super::super::super::gguf_host::source_copy_output_control_bytes(
                    &plan,
                    &self.host_runtime,
                    ordinal,
                )
                .unwrap()
            })
            .collect()
    }
    /// Actual combined amount charged at each reached source construction.
    pub(crate) fn source_storage_layouts(&self) -> Vec<u64> {
        let plan = self.source.conversion_plan(&Self::request()).unwrap();
        (0..plan.conversion().outputs().len())
            .map(|ordinal| {
                super::super::super::gguf_host::source_copy_output_storage_bytes(
                    &plan,
                    &self.host_runtime,
                    ordinal,
                )
                .unwrap()
            })
            .collect()
    }
    fn source_ready(
        &self,
        controls: &OriginalTextControlGuard,
        bank: OriginalHostSourceBank,
    ) -> PreparedPendingWeight {
        PreparedPendingWeight::new_with_gguf_source_constructions(
            controls.clone(),
            Rc::clone(&self.host_runtime),
            bank,
        )
        .unwrap_or_else(|(_, cause)| panic!("source bank: {cause}"))
    }
    pub(crate) fn funded_source_miss_hit_miss(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        mut bank: OriginalHostSourceBank,
    ) -> Array {
        let layouts = self.source_storage_layouts();
        let bytes: u64 = layouts.iter().sum();
        let attempts = layouts.len();
        let before = self.source.diagnostics().unwrap().physical_reads;
        let first = self.lease();
        let address = Self::address(&first);
        let first = first
            .prepare_original_gguf_fixture(
                &self.stream,
                self.source_ready(controls, bank.split(bytes, attempts).unwrap()),
                observer.clone(),
            )
            .unwrap();
        assert_eq!(Self::address(first.lease()), address);
        self.settle(&first, observer);
        let first_host = first.retained.retention().gguf_host.as_ref().unwrap();
        let remaining = first_host.source_constructions.as_ref().unwrap();
        assert_eq!(
            (remaining.remaining_bytes(), remaining.remaining_attempts()),
            (0, 0)
        );
        assert_eq!(first_host.facts().iter().flatten().count(), attempts);
        let group = first.retained.retention().group.as_ref().unwrap().clone();
        for (name, expected) in &self.expected {
            assert_eq!(
                &super::values(group.arrays.get(name).unwrap(), Some(observer)),
                expected
            );
        }
        let second = self
            .lease()
            .prepare_original_gguf_fixture(
                &self.stream,
                self.source_ready(controls, bank.split(bytes, attempts).unwrap()),
                observer.clone(),
            )
            .unwrap();
        let second_host = second.retained.retention().gguf_host.as_ref().unwrap();
        assert!(second_host.facts().iter().all(Option::is_none));
        let remaining = second_host.source_constructions.as_ref().unwrap();
        assert_eq!(
            (remaining.remaining_bytes(), remaining.remaining_attempts()),
            (bytes, attempts)
        );
        assert!(second
            .retained
            .retention()
            .group
            .as_ref()
            .unwrap()
            .same(&group));
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 1
        );
        self.settle(&second, observer);
        let weak = group.downgrade();
        drop(group);
        Self::finish(second);
        Self::finish(first);
        assert!(weak.upgrade().is_none());
        let third = self
            .lease()
            .prepare_original_gguf_fixture(
                &self.stream,
                self.source_ready(controls, bank.split(bytes, attempts).unwrap()),
                observer.clone(),
            )
            .unwrap();
        assert_eq!(
            (
                bank.remaining_bytes(),
                bank.remaining_attempts(),
                bank.remaining_partitions()
            ),
            (0, 0, 0)
        );
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 2
        );
        self.settle(&third, observer);
        let output = third.output().clone();
        Self::finish(third);
        output
    }
    pub(crate) fn funded_source_refusal(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        bank: OriginalHostSourceBank,
        ordinal: usize,
    ) -> PreparedGgufHostCopyFailure {
        let available = bank.remaining_bytes();
        let layouts = self.source_storage_layouts();
        let lease = self.lease();
        let address = Self::address(&lease);
        let mut pending = PendingWeightMaterialization::begin_with_original(
            lease,
            &self.stream,
            &self.stream,
            Some((self.source_ready(controls, bank), observer.clone())),
        )
        .unwrap();
        let portable = pending.materialize_gguf_prepared().unwrap();
        let error = pending.convert_original_gguf(portable).unwrap_err();
        assert_eq!(error.output_ordinal(), ordinal);
        let GgufHostCopyCause::SourceFunding(cause) = error.cause() else {
            panic!("source refusal: {error:?}")
        };
        assert!(matches!(
            cause.cause(),
            HostDestinationCause::Capacity { required, remaining }
                if *required == layouts[ordinal]
                    && *remaining == available - layouts[..ordinal].iter().sum::<u64>()
        ));
        assert!(cause.retains_receipt());
        assert_eq!(Self::address(pending.lease()), address);
        let host = pending.retained.retention().gguf_host.as_ref().unwrap();
        host.assert_failed_input_unchanged();
        host.assert_source_refusal_precedes_arena();
        assert_eq!(host.facts().iter().flatten().count(), ordinal + 1);
        let bank = host.source_constructions.as_ref().unwrap();
        assert_eq!(bank.remaining_attempts(), 0);
        assert!(pending.retained.retention().group.is_none());
        assert!(observer.status().is_settled() && !observer.status().failed());
        Self::finish(pending);
        safemlx::reclaim_allocation_owners();
        error
    }
    pub(crate) fn reject_foreign_source_bank(
        &self,
        controls: &OriginalTextControlGuard,
        bank: OriginalHostSourceBank,
    ) -> OriginalHostSourceBank {
        let before = (
            bank.remaining_bytes(),
            bank.remaining_attempts(),
            bank.remaining_partitions(),
        );
        let (bank, error) = PreparedPendingWeight::new_with_gguf_source_constructions(
            controls.clone(),
            Rc::clone(&self.host_runtime),
            bank,
        )
        .err()
        .expect("foreign source bank must refuse");
        assert!(matches!(
            error,
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch
        ));
        assert_eq!(
            (
                bank.remaining_bytes(),
                bank.remaining_attempts(),
                bank.remaining_partitions()
            ),
            before
        );
        bank
    }
}

impl OriginalGgufMissFixture {
    pub(crate) fn funded_destinations_miss_hit_miss(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        mut bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
    ) -> Array {
        let layouts = self.source_storage_layouts();
        let bytes = layouts.iter().sum::<u64>()
            + super::super::super::super::cache::control_bytes().unwrap();
        let attempts = layouts.len() + 1;
        let destination = PreparedPendingWeight::host_destination_requests(
            &self.source.conversion_plan(&Self::request()).unwrap(),
        )
        .unwrap();
        let destination_bytes = u64::try_from(destination.bytes()).unwrap();
        let before = self.source.diagnostics().unwrap().physical_reads;
        let first = self.lease();
        let address = Self::address(&first);
        let first = first
            .prepare_original_gguf_fixture(
                &self.stream,
                self.admitted_ready(
                    controls,
                    bank.split(
                        destination_bytes,
                        destination.calls(),
                        Some((bytes, attempts)),
                    )
                    .unwrap(),
                ),
                observer.clone(),
            )
            .unwrap();
        assert_eq!(Self::address(first.lease()), address);
        self.settle(&first, observer);
        let first_host = first.retained.retention().gguf_host.as_ref().unwrap();
        let remaining = first_host.source_constructions.as_ref().unwrap();
        assert_eq!(
            (remaining.remaining_bytes(), remaining.remaining_attempts()),
            (0, 0)
        );
        let first_bank = first_host.host_destinations.as_ref().unwrap();
        assert_eq!(
            (
                first_bank.remaining_bytes(),
                first_bank.remaining_attempts()
            ),
            (0, 0)
        );
        assert_eq!(first_host.facts().iter().flatten().count(), layouts.len());
        let group = first.retained.retention().group.as_ref().unwrap().clone();
        for (name, expected) in &self.expected {
            assert_eq!(
                &super::values(group.arrays.get(name).unwrap(), Some(observer)),
                expected
            );
        }
        let second = self
            .lease()
            .prepare_original_gguf_fixture(
                &self.stream,
                self.admitted_ready(
                    controls,
                    bank.split(
                        destination_bytes,
                        destination.calls(),
                        Some((bytes, attempts)),
                    )
                    .unwrap(),
                ),
                observer.clone(),
            )
            .unwrap();
        let second_host = second.retained.retention().gguf_host.as_ref().unwrap();
        assert!(second_host.facts().iter().all(Option::is_none));
        let second_bank = second_host.host_destinations.as_ref().unwrap();
        assert_eq!(
            (
                second_bank.remaining_bytes(),
                second_bank.remaining_attempts()
            ),
            (destination_bytes, destination.calls())
        );
        let remaining = second_host.source_constructions.as_ref().unwrap();
        assert_eq!(
            (remaining.remaining_bytes(), remaining.remaining_attempts()),
            (bytes, attempts)
        );
        assert!(second
            .retained
            .retention()
            .group
            .as_ref()
            .unwrap()
            .same(&group));
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 1
        );
        self.settle(&second, observer);
        let weak = group.downgrade();
        drop(group);
        Self::finish(second);
        Self::finish(first);
        assert!(weak.upgrade().is_none());
        let third = self
            .lease()
            .prepare_original_gguf_fixture(
                &self.stream,
                self.admitted_ready(
                    controls,
                    bank.split(
                        destination_bytes,
                        destination.calls(),
                        Some((bytes, attempts)),
                    )
                    .unwrap(),
                ),
                observer.clone(),
            )
            .unwrap();
        assert_eq!(
            (
                bank.remaining_bytes(),
                bank.remaining_attempts(),
                bank.remaining_partitions()
            ),
            (0, 0, 0)
        );
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 2
        );
        self.settle(&third, observer);
        let output = third.output().clone();
        Self::finish(third);
        output
    }
}

impl OriginalGgufMissFixture {
    pub(crate) fn funded_destinations_refusal(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
        ordinal: usize,
    ) -> PreparedGgufHostCopyFailure {
        let layouts = self.source_storage_layouts();
        let lease = self.lease();
        let address = Self::address(&lease);
        let mut pending = PendingWeightMaterialization::begin_with_original(
            lease,
            &self.stream,
            &self.stream,
            Some((self.admitted_ready(controls, bank), observer.clone())),
        )
        .unwrap();
        let portable = pending.materialize_gguf_prepared().unwrap();
        let error = pending.convert_original_gguf(portable).unwrap_err();
        assert_eq!(error.output_ordinal(), ordinal);
        let GgufHostCopyCause::SourceFunding(cause) = error.cause() else {
            panic!("source refusal: {error:?}")
        };
        assert!(matches!(
            cause.cause(),
            HostDestinationCause::Capacity { required, .. }
                if *required == layouts[ordinal]
        ));
        assert!(cause.retains_receipt());
        assert_eq!(Self::address(pending.lease()), address);
        let host = pending.retained.retention().gguf_host.as_ref().unwrap();
        host.assert_failed_input_unchanged();
        host.assert_source_refusal_precedes_arena();
        assert_eq!(host.facts().iter().flatten().count(), ordinal + 1);
        let bank = host.source_constructions.as_ref().unwrap();
        assert_eq!(bank.remaining_attempts(), 0);
        assert!(pending.retained.retention().group.is_none());
        assert!(observer.status().is_settled() && !observer.status().failed());
        Self::finish(pending);
        safemlx::reclaim_allocation_owners();
        error
    }
}
