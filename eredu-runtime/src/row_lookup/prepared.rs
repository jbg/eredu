//! Retained row sources and exact companions for typed native binding.
use super::*;
use crate::RowResidencyRange;
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{PreparedCheckpointSource, PreparedTensorSource, SharedCheckpointSource},
};

/// One retained floating scalar, separate from the row payload residency.
#[derive(Clone)]
pub struct PreparedRowScale {
    /// Exact companion identity; native code must not derive its name.
    pub parameter: ParameterId,
    /// Retained, restricted source and provenance.
    pub source: SharedCheckpointSource,
    /// Exact scalar recipe.
    pub recipe: DerivedWeightRecipe,
}
impl std::fmt::Debug for PreparedRowScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedRowScale")
            .field("parameter", &self.parameter)
            .field("recipe", &self.recipe)
            .finish_non_exhaustive()
    }
}

/// One compact row bank's complete binding contract. No entry per row is created.
#[derive(Debug, Clone)]
pub struct PreparedRowLookup {
    range: RowResidencyRange,
    descriptor: RowLookupDescriptor,
    scale: Option<PreparedRowScale>,
}
// Equality is retained binding identity, never equality of source shapes or names.
impl PartialEq for PreparedRowLookup {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
            && self.range.same_binding(&other.range)
            && match (&self.scale, &other.scale) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.parameter == b.parameter
                        && a.recipe == b.recipe
                        && std::sync::Arc::ptr_eq(&a.source, &b.source)
                }
                _ => false,
            }
    }
}
impl PreparedRowLookup {
    /// Checks source geometry, companions and maximum request workspace before
    /// native construction or payload acquisition.
    pub fn new(
        range: RowResidencyRange,
        spec: RowLookupSpec,
        mut scale: Option<PreparedRowScale>,
        limits: RowLookupLimits,
    ) -> Result<Self, RowLookupError> {
        if let Some(scale) = &mut scale {
            let catalog = scale.source.source_keys().into_iter().map(|key| {
                Ok((key.clone(), PreparedTensorSource {
                    metadata: scale.source.source_metadata(&key)?,
                    provenance: scale.source.source_provenance(&key)?,
                }))
            }).collect::<Result<std::collections::BTreeMap<_, _>, eredu_checkpoint::store::StoreError>>()
                .map_err(RowLookupError::bank)?;
            scale.source = std::sync::Arc::new(
                PreparedCheckpointSource::new(scale.source.clone(), catalog)
                    .map_err(RowLookupError::bank)?,
            );
        }
        let descriptor = RowLookupDescriptor::new(
            range.range().clone(),
            range.metadata().clone(),
            spec,
            scale
                .as_ref()
                .map(|scale| {
                    let catalog = scale
                        .source
                        .source_keys()
                        .into_iter()
                        .map(|key| {
                            Ok((
                                key.clone(),
                                RowScaleSource {
                                    metadata: scale.source.source_metadata(&key)?,
                                    encoding: scale.source.source_provenance(&key)?.source_encoding,
                                },
                            ))
                        })
                        .collect::<Result<_, eredu_checkpoint::store::StoreError>>()
                        .map_err(RowLookupError::bank)?;
                    RowScaleDescriptor::new(scale.parameter.clone(), scale.recipe.clone(), catalog)
                })
                .transpose()?,
            limits,
        )?;
        Ok(Self {
            range,
            descriptor,
            scale,
        })
    }
    /// Metadata-only contract used by cold mechanism admission.
    pub fn descriptor(&self) -> &RowLookupDescriptor {
        &self.descriptor
    }
    /// Exact compact residency range, including its retained source.
    pub fn range(&self) -> &RowResidencyRange {
        &self.range
    }
    /// Exact operator geometry and ownership.
    pub fn spec(&self) -> &RowLookupSpec {
        self.descriptor.spec()
    }
    /// Exact scalar companion, when selected.
    pub fn scale(&self) -> Option<&PreparedRowScale> {
        self.scale.as_ref()
    }
    /// Retained request and workspace bounds.
    pub fn limits(&self) -> RowLookupLimits {
        self.descriptor.limits()
    }
    /// Largest distinct-row acquisition reachable under every retained limit.
    /// This depends on metadata only, never vocabulary-sized catalog expansion.
    pub fn maximum_acquisition_rows(&self) -> u64 {
        self.descriptor.maximum_acquisition_rows()
    }
    /// Pairs a native bank with the admitted portable provider. Native construction
    /// consumes this same contract; no names, formats or bounds are reconstructed.
    pub fn bind<K>(&self, bank: K) -> Result<BoundedRowLookup<K>, RowLookupError> {
        BoundedRowLookup::new(bank, self.spec().clone(), self.limits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{
        rows::PreparedRowSource,
        store::{MemoryWeightStore, TensorSelection},
    };
    use eredu_core::residency::{OffloadUnitId, OffloadUnitRange, ResidencyPolicy};
    use std::sync::Arc;
    pub(super) fn fixture(
        fp8: bool,
    ) -> (
        RowResidencyRange,
        RowLookupSpec,
        Option<PreparedRowScale>,
        RowLookupLimits,
    ) {
        let dtype = if fp8 {
            safetensors::Dtype::F8_E4M3
        } else {
            safetensors::Dtype::F32
        };
        let row_bytes = if fp8 { 2 } else { 8 };
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "rows".into(),
                dtype,
                vec![4, 2],
                vec![0; row_bytes * 4],
            )])
            .unwrap(),
        );
        let table = PreparedRowSource::new(
            source,
            DerivedWeightRecipe::source("rows", TensorSelection::Full),
        )
        .unwrap();
        let range = RowResidencyRange::new(
            OffloadUnitRange::new(
                OffloadUnitId::new("bank.rows").unwrap(),
                0,
                4,
                row_bytes as u64,
                ResidencyPolicy::Cacheable,
            )
            .unwrap(),
            table,
            "table",
        )
        .unwrap();
        let scale = fp8.then(|| PreparedRowScale {
            parameter: ParameterId::new("exact.scalar").unwrap(),
            source: Arc::new(
                MemoryWeightStore::from_safetensors([(
                    "scalar".into(),
                    safetensors::Dtype::F32,
                    vec![1],
                    2f32.to_le_bytes().to_vec(),
                )])
                .unwrap(),
            ),
            recipe: DerivedWeightRecipe::source("scalar", TensorSelection::Full),
        });
        let spec = RowLookupSpec {
            parameter: ParameterId::new("table").unwrap(),
            bank: 1,
            unit: 3,
            rows: 4,
            dimensions: 2,
            encoding: scale
                .as_ref()
                .map_or(RowEncoding::Dense, |s| RowEncoding::ScalarE4M3 {
                    scale: s.parameter.clone(),
                }),
            output_type: TensorElementType::F32,
        };
        let limits = RowLookupLimits {
            requests: 8,
            rows_per_acquisition: 2,
            acquisition_bytes: 16,
            host_bytes: 1024,
            output_bytes: 192,
        };
        (range, spec, scale, limits)
    }
    #[test]
    fn prepared_rows_admit_workspace_and_companions_without_payload_reads() {
        for fp8 in [false, true] {
            let (range, spec, scale, limits) = fixture(fp8);
            let prepared =
                PreparedRowLookup::new(range.clone(), spec.clone(), scale.clone(), limits).unwrap();
            assert!(prepared.range().same_binding(&range));
            assert!(!prepared.range().same_binding(&fixture(fp8).0));
            assert_eq!(
                prepared
                    .range()
                    .source()
                    .source_diagnostics()
                    .unwrap()
                    .physical_read_bytes,
                0
            );
            if let Some(scale) = prepared.scale() {
                assert_eq!(
                    scale
                        .source
                        .source_diagnostics()
                        .unwrap()
                        .physical_read_bytes,
                    0
                );
            }
            let mut small = limits;
            small.host_bytes -= 1;
            assert!(matches!(
                PreparedRowLookup::new(range.clone(), spec.clone(), scale.clone(), small),
                Err(RowLookupError::Budget {
                    resource: "host planning bytes",
                    ..
                })
            ));
            small = limits;
            small.output_bytes -= 1;
            assert!(matches!(
                PreparedRowLookup::new(range.clone(), spec.clone(), scale.clone(), small),
                Err(RowLookupError::Budget {
                    resource: "output bytes",
                    ..
                })
            ));
            small = limits;
            small.acquisition_bytes = 1;
            assert!(matches!(
                PreparedRowLookup::new(range.clone(), spec.clone(), scale.clone(), small),
                Err(RowLookupError::Budget {
                    resource: "acquisition bytes",
                    ..
                })
            ));
            let mut bad = spec.clone();
            bad.rows += 1;
            assert!(PreparedRowLookup::new(range.clone(), bad, scale.clone(), limits).is_err());
            if fp8 {
                assert!(PreparedRowLookup::new(range.clone(), spec.clone(), None, limits).is_err());
                let mut wrong = scale.unwrap();
                wrong.parameter = ParameterId::new("wrong").unwrap();
                assert!(PreparedRowLookup::new(range, spec, Some(wrong), limits).is_err());
            }
        }
    }
}

#[cfg(test)]
mod collection_tests {
    use super::*;
    use std::collections::BTreeMap;
    #[test]
    fn prepared_row_set_checks_owners_exact_binding_and_cold_totals() {
        for fp8 in [false, true] {
            let (range, spec, scale, limits) = super::tests::fixture(fp8);
            let entry = PreparedRowLookup::new(range, spec, scale, limits).unwrap();
            let id = entry.spec().parameter.clone();
            let rows = PreparedRowLookups::new([entry.clone()], 4).unwrap();
            assert_eq!(rows.entries().len(), 1);
            assert_eq!(rows.ranges().count(), 1);
            assert_eq!(
                rows.requirements(),
                RowLookupRequirements {
                    source_bytes: if fp8 { 8 } else { 32 },
                    scalar_bytes: if fp8 { 4 } else { 0 },
                    acquisition_bytes: if fp8 { 4 } else { 16 },
                    host_bytes: 1024,
                    output_bytes: 192,
                }
            );
            assert!(PreparedRowLookups::new([entry.clone()], 3).is_err());
            assert!(PreparedRowLookups::new([entry.clone(), entry.clone()], 4).is_err());
            assert!(rows.validate_other_banks([1]).is_err());
            rows.validate_other_banks([0, 2]).unwrap();
            let required = rows.requirements().acquisition_bytes;
            rows.validate_pool_budget(
                eredu_core::residency::OffloadConfig::new(Some(required), None, 1).unwrap(),
            )
            .unwrap();
            assert!(matches!(
                rows.validate_pool_budget(
                    eredu_core::residency::OffloadConfig::new(Some(required - 1), None, 1).unwrap()
                ),
                Err(RowLookupError::Budget {
                    resource: "row acquisition residency",
                    ..
                })
            ));
            assert!(matches!(
                rows.bind(BTreeMap::<ParameterId, ()>::new()),
                Err(RowLookupError::Missing(_))
            ));
            assert!(matches!(
                rows.bind(BTreeMap::from([
                    (id.clone(), ()),
                    (ParameterId::new("extra").unwrap(), ())
                ])),
                Err(RowLookupError::Specification(_))
            ));
            assert_eq!(
                rows.bind(BTreeMap::from([(id, ())]))
                    .unwrap()
                    .providers()
                    .len(),
                1
            );
            assert_eq!(
                entry
                    .range()
                    .source()
                    .source_diagnostics()
                    .unwrap()
                    .physical_read_bytes,
                0
            );
            if let Some(scale) = entry.scale() {
                assert_eq!(
                    scale
                        .source
                        .source_diagnostics()
                        .unwrap()
                        .physical_read_bytes,
                    0
                );
            }
        }
        assert_eq!(
            PreparedRowLookups::new([], 0).unwrap().requirements(),
            RowLookupRequirements::default()
        );
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    include!("memory_tests.rs");
    use crate::{
        AddressableStorageCapabilities, AddressableStorageTiers, ParameterBankLoadOptions,
    };
    use eredu_core::residency::OffloadConfig;

    struct Support {
        storage: Option<AddressableStorageCapabilities>,
        workspace: Option<RowLookupWorkspace>,
    }
    impl Default for Support {
        fn default() -> Self {
            Self {
                storage: Some(AddressableStorageCapabilities::new(
                    true,
                    true,
                    true,
                    u64::MAX,
                )),
                workspace: Some(RowLookupWorkspace {
                    decode_bytes: 80,
                    scalar_bytes: 4,
                }),
            }
        }
    }
    impl RowLookupMechanismSupport for Support {
        fn storage(&self) -> Option<AddressableStorageCapabilities> {
            self.storage
        }
        fn workspace(
            &self,
            entry: &RowLookupDescriptor,
        ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
            Ok(self.workspace.map(|w| RowLookupWorkspace {
                scalar_bytes: if entry.scale().is_some() {
                    w.scalar_bytes
                } else {
                    0
                },
                ..w
            }))
        }
    }
    fn prepared(fp8: bool) -> PreparedRowLookups {
        let (range, spec, scale, limits) = super::tests::fixture(fp8);
        PreparedRowLookups::new(
            [PreparedRowLookup::new(range, spec, scale, limits).unwrap()],
            4,
        )
        .unwrap()
    }
    fn options(device: u64, scratch: u64) -> ParameterBankLoadOptions {
        ParameterBankLoadOptions::new(
            OffloadConfig::new(Some(device), Some(0), 1).unwrap(),
            scratch,
            scratch,
        )
        .unwrap()
    }
    fn assert_lazy(rows: &PreparedRowLookups) {
        for entry in rows.entries().values() {
            assert_eq!(
                entry
                    .range()
                    .source()
                    .source_diagnostics()
                    .unwrap()
                    .physical_read_bytes,
                0
            );
            if let Some(scale) = entry.scale() {
                assert_eq!(
                    scale
                        .source
                        .source_diagnostics()
                        .unwrap()
                        .physical_read_bytes,
                    0
                );
            }
        }
    }
    #[test]
    fn cold_row_selection_retains_sources_and_combines_simultaneous_workspace() {
        for fp8 in [false, true] {
            let prepared = prepared(fp8);
            let total = 1024 + 192 + 80;
            let selected = SelectedRowLookups::select(
                prepared.clone(),
                options(16, total),
                4,
                &Support::default(),
            )
            .unwrap();
            assert_eq!(selected.requirements().invocation_bytes, total);
            assert_eq!(
                selected.requirements().scalar_preparation_bytes,
                if fp8 { 4 } else { 0 }
            );
            assert_eq!(selected.requirements().rows, prepared.requirements());
            let original = prepared.entries().values().next().unwrap();
            let retained = selected.prepared().entries().values().next().unwrap();
            assert!(original.range().same_binding(retained.range()));
            if fp8 {
                assert!(std::sync::Arc::ptr_eq(
                    &original.scale().unwrap().source,
                    &retained.scale().unwrap().source
                ));
            }
            selected
                .validate_binding(options(16, total).offload(), &Support::default())
                .unwrap();
            assert!(
                matches!(SelectedRowLookups::select(prepared.clone(), options(16, total - 1), 4, &Support::default()),
                Err(RowLookupSelectionError::Lookup(RowLookupError::Budget { resource: "row workspace", required, .. })) if required == total)
            );
            assert!(matches!(
                selected.validate_binding(options(17, total).offload(), &Support::default()),
                Err(RowLookupSelectionError::PoolMismatch)
            ));
            let changed = Support {
                workspace: Some(RowLookupWorkspace {
                    decode_bytes: 79,
                    scalar_bytes: 4,
                }),
                ..Default::default()
            };
            assert!(matches!(
                selected.validate_binding(options(16, total).offload(), &changed),
                Err(RowLookupSelectionError::Mechanism { .. })
            ));
            assert_lazy(&prepared);
        }
    }
    #[test]
    fn cold_row_selection_rejects_missing_mechanisms_and_resource_overflow() {
        let rows = prepared(true);
        let policy = options(4, 4096);
        for storage in [
            None,
            Some(AddressableStorageCapabilities::new(
                false,
                true,
                true,
                u64::MAX,
            )),
            Some(AddressableStorageCapabilities::new(
                true,
                false,
                true,
                u64::MAX,
            )),
            Some(AddressableStorageCapabilities::new(
                true,
                true,
                false,
                u64::MAX,
            )),
            Some(
                AddressableStorageCapabilities::new(true, true, true, u64::MAX)
                    .with_tiers(AddressableStorageTiers::new(false, true, true)),
            ),
            Some(
                AddressableStorageCapabilities::new(true, true, true, u64::MAX)
                    .with_tiers(AddressableStorageTiers::new(true, true, false)),
            ),
        ] {
            assert!(matches!(
                SelectedRowLookups::select(
                    rows.clone(),
                    policy,
                    4,
                    &Support {
                        storage,
                        ..Default::default()
                    }
                ),
                Err(RowLookupSelectionError::Mechanism { .. })
            ));
        }
        let no_host = Support {
            storage: Some(
                AddressableStorageCapabilities::new(true, true, true, 4096)
                    .with_tiers(AddressableStorageTiers::new(true, false, true)),
            ),
            ..Default::default()
        };
        SelectedRowLookups::select(rows.clone(), policy, 4, &no_host).unwrap();
        let with_host = ParameterBankLoadOptions::new(
            OffloadConfig::new(Some(4), Some(16), 1).unwrap(),
            4096,
            4096,
        )
        .unwrap();
        assert!(matches!(
            SelectedRowLookups::select(rows.clone(), with_host, 4, &no_host),
            Err(RowLookupSelectionError::Mechanism {
                mechanism: "host row storage",
                ..
            })
        ));
        assert!(matches!(
            SelectedRowLookups::select(rows.clone(), options(3, 4096), 4, &Support::default()),
            Err(RowLookupSelectionError::Lookup(RowLookupError::Budget {
                resource: "row acquisition residency",
                ..
            }))
        ));
        assert!(matches!(
            SelectedRowLookups::select(rows.clone(), policy, 3, &Support::default()),
            Err(RowLookupSelectionError::Lookup(RowLookupError::Budget {
                resource: "retained row scalars",
                ..
            }))
        ));
        assert!(matches!(
            SelectedRowLookups::select(
                rows.clone(),
                policy,
                4,
                &Support {
                    workspace: None,
                    ..Default::default()
                }
            ),
            Err(RowLookupSelectionError::Mechanism {
                parameter: Some(_),
                ..
            })
        ));
        for workspace in [
            RowLookupWorkspace {
                decode_bytes: 0,
                scalar_bytes: 4,
            },
            RowLookupWorkspace {
                decode_bytes: 80,
                scalar_bytes: 0,
            },
            RowLookupWorkspace {
                decode_bytes: u64::MAX,
                scalar_bytes: 4,
            },
        ] {
            assert!(matches!(
                SelectedRowLookups::select(
                    rows.clone(),
                    policy,
                    4,
                    &Support {
                        workspace: Some(workspace),
                        ..Default::default()
                    }
                ),
                Err(RowLookupSelectionError::Lookup(RowLookupError::Geometry))
            ));
        }
        let small_backend = Support {
            storage: Some(AddressableStorageCapabilities::new(true, true, true, 1295)),
            ..Default::default()
        };
        assert!(matches!(
            SelectedRowLookups::select(rows.clone(), policy, 4, &small_backend),
            Err(RowLookupSelectionError::Lookup(RowLookupError::Budget {
                resource: "backend row workspace",
                ..
            }))
        ));
        let large_recipe = Support {
            workspace: Some(RowLookupWorkspace {
                decode_bytes: 80,
                scalar_bytes: 4097,
            }),
            ..Default::default()
        };
        assert!(matches!(
            SelectedRowLookups::select(rows.clone(), policy, 4, &large_recipe),
            Err(RowLookupSelectionError::Lookup(RowLookupError::Budget {
                required: 4097,
                ..
            }))
        ));
        assert_lazy(&rows);
    }
    #[test]
    fn row_selection_sums_retained_scalars_but_peaks_each_serial_invocation() {
        use eredu_checkpoint::{
            rows::PreparedRowSource,
            store::{MemoryWeightStore, TensorSelection},
        };
        use eredu_core::residency::{OffloadUnitId, OffloadUnitRange, ResidencyPolicy};
        // Each binding owns a scalar even when both refer to the same source.
        let original = prepared(true);
        let first = original.entries().values().next().unwrap().clone();
        let mut spec = first.spec().clone();
        spec.parameter = ParameterId::new("second.table").unwrap();
        spec.bank = 2;
        spec.unit = 2;
        let source = std::sync::Arc::new(
            MemoryWeightStore::from_safetensors([(
                "other".into(),
                safetensors::Dtype::F8_E4M3,
                vec![4, 2],
                vec![0; 8],
            )])
            .unwrap(),
        );
        let range = RowResidencyRange::new(
            OffloadUnitRange::new(
                OffloadUnitId::new("other.rows").unwrap(),
                0,
                4,
                2,
                ResidencyPolicy::Cacheable,
            )
            .unwrap(),
            PreparedRowSource::new(
                source,
                DerivedWeightRecipe::source("other", TensorSelection::Full),
            )
            .unwrap(),
            "rows",
        )
        .unwrap();
        let second = PreparedRowLookup::new(
            range,
            spec,
            first.scale().cloned(),
            RowLookupLimits {
                requests: 4,
                host_bytes: 512,
                output_bytes: 96,
                ..first.limits()
            },
        )
        .unwrap();
        let rows = PreparedRowLookups::new([first, second], 4).unwrap();
        struct PerTable;
        impl RowLookupMechanismSupport for PerTable {
            fn storage(&self) -> Option<AddressableStorageCapabilities> {
                Support::default().storage()
            }
            fn workspace(
                &self,
                entry: &RowLookupDescriptor,
            ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
                Ok(Some(RowLookupWorkspace {
                    decode_bytes: if entry.spec().bank == 2 { 200 } else { 80 },
                    scalar_bytes: 4,
                }))
            }
        }
        let selected =
            SelectedRowLookups::select(rows.clone(), options(4, 1296), 8, &PerTable).unwrap();
        assert_eq!(selected.requirements().rows.scalar_bytes, 8);
        assert_eq!(selected.requirements().rows.source_bytes, 16);
        assert_eq!(selected.requirements().decode_bytes, 200);
        assert_eq!(selected.requirements().invocation_bytes, 1296);
        // Pipeline ownership filters retained mechanisms and source authority by
        // global unit identity without consulting the backend a second time.
        let stage = selected.plan().clone().for_units(3..4, 4).unwrap();
        assert_eq!(stage.options(), selected.options());
        assert_eq!(stage.descriptors().entries().len(), 1);
        assert_eq!(stage.requirements().rows.scalar_bytes, 4);
        assert_eq!(stage.requirements().rows.source_bytes, 8);
        assert_eq!(stage.requirements().decode_bytes, 80);
        let (id, entry) = stage.descriptors().entries().first_key_value().unwrap();
        assert_eq!(entry.spec().unit, 3);
        assert_eq!(stage.workspace(id), selected.plan().workspace(id));
        assert_eq!(stage.decode_memory().len(), 1);
        assert_eq!(
            stage.decode_memory()[id],
            selected.plan().decode_memory()[id]
        );
        let stage_sources = PreparedRowLookups::new(
            rows.entries()
                .values()
                .filter(|entry| entry.spec().unit == 3)
                .cloned(),
            4,
        )
        .unwrap();
        assert!(stage.clone().bind(rows.clone()).is_err());
        assert_eq!(
            stage.bind(stage_sources.clone()).unwrap().prepared(),
            &stage_sources
        );
        let empty = selected.plan().clone().for_units(0..2, 4).unwrap();
        assert_eq!(
            empty.requirements(),
            SelectedRowLookupRequirements::default()
        );
        assert!(empty.descriptors().entries().is_empty());
        assert!(empty.decode_memory().is_empty());
        assert!(empty.bind(PreparedRowLookups::default()).is_ok());
        for range in [3..2, 0..5] {
            assert!(selected.plan().clone().for_units(range, 4).is_err());
        }
        assert!(matches!(
            SelectedRowLookups::select(rows.clone(), options(4, 1296), 7, &PerTable),
            Err(RowLookupSelectionError::Lookup(RowLookupError::Budget {
                resource: "retained row scalars",
                required: 8,
                ..
            }))
        ));
        assert_lazy(&rows);
    }
    #[test]
    fn empty_row_selection_needs_no_row_mechanism() {
        let selected = SelectedRowLookups::select(
            PreparedRowLookups::default(),
            ParameterBankLoadOptions::default(),
            0,
            &Support {
                storage: None,
                workspace: None,
            },
        )
        .unwrap();
        assert_eq!(
            selected.requirements(),
            SelectedRowLookupRequirements::default()
        );
    }
}
