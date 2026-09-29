use eredu_nn::mechanism_memory::*;

struct MemorySupport {
    mode: u8,
}
impl RowLookupMechanismSupport for MemorySupport {
    fn storage(&self) -> Option<AddressableStorageCapabilities> {
        Support::default().storage()
    }
    fn workspace(
        &self,
        entry: &RowLookupDescriptor,
    ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
        Support::default().workspace(entry)
    }
    fn decode_memory(
        &self,
        entry: &RowLookupDescriptor,
    ) -> Result<MechanismMemoryContract, RowLookupError> {
        let mut memory = entry.decode_memory()?;
        if self.mode == 1 {
            memory.values.last_mut().unwrap().shape[1] += 1;
        }
        let bytes = memory.values.last().unwrap().logical_bytes()?;
        memory.storage.push(MechanismStorage {
            name: "output".into(),
            role: MechanismStorageRole::Output,
            payload: MechanismBytes::exact(bytes),
            capacity: if self.mode == 2 {
                MechanismBytes::exact(0)
            } else {
                MechanismBytes::unknown(bytes)
            },
            backing: MechanismBacking::Invocation,
            placement: MechanismPlacement::Execution,
            retention: if self.mode == 3 {
                StorageRetention::NativeCompletion
            } else {
                StorageRetention::Returned
            },
            detail: "independent neutral fixture output backing".into(),
        });
        Ok(memory)
    }
}
#[test]
fn row_decoder_memory_is_retained_validated_and_bound_without_payload_reads() {
    use eredu_core::{resources::*, Observed};
    for fp8 in [false, true] {
        let rows = prepared(fp8);
        let entry = rows.entries().values().next().unwrap();
        let id = &entry.spec().parameter;
        let logical = entry.descriptor().decode_memory().unwrap();
        assert_eq!(logical.values.len(), if fp8 { 3 } else { 2 });
        assert_eq!(
            logical.values.last().unwrap().shape,
            [entry.maximum_acquisition_rows(), 2]
        );
        assert_eq!(
            logical.values[0].logical_bytes().unwrap(),
            entry.maximum_acquisition_rows() * entry.range().range().bytes()
        );
        assert!(logical.storage.is_empty() && !logical.missing.is_empty());
        let policy = options(16, 4096);
        for mode in [1, 2] {
            assert!(
                SelectedRowLookups::select(rows.clone(), policy, 4, &MemorySupport { mode })
                    .is_err()
            );
        }
        let selected =
            SelectedRowLookups::select(rows.clone(), policy, 4, &MemorySupport { mode: 0 })
                .unwrap();
        selected
            .validate_binding(policy.offload(), &MemorySupport { mode: 0 })
            .unwrap();
        assert!(matches!(
            selected.validate_binding(policy.offload(), &MemorySupport { mode: 3 }),
            Err(RowLookupSelectionError::Mechanism {
                mechanism: "selected row decoder storage or retention",
                ..
            })
        ));
        let query = crate::MechanismResourceQuery {
            invocation: ResourceIdentity {
                scope: "fixture".into(),
                key: "lookup".into(),
            },
            execution_pool: Observed::Unavailable {
                reason: "no native device".into(),
            },
            host_pool: Observed::Unavailable {
                reason: "no host pool binding".into(),
            },
            owner_backings: Default::default(),
        };
        let described = selected.describe_decode_resources(id, &query).unwrap();
        assert_eq!(described.contract, selected.decode_memory()[id]);
        assert_eq!(described.resources.allocations.len(), 1);
        assert!(matches!(
            described.resources.coverage,
            ResourceCoverage::Partial { .. }
        ));
        assert!(selected
            .describe_decode_resources(&ParameterId::new("unknown").unwrap(), &query)
            .is_err());
        let mut other = query.clone();
        other.invocation.key = "next-lookup".into();
        assert_ne!(
            described.resources.allocations[0].identity,
            selected
                .describe_decode_resources(id, &other)
                .unwrap()
                .resources
                .allocations[0]
                .identity
        );
        assert_lazy(&rows);
    }
}
