use super::*;
use crate::{PreparationMechanismProvider, PreparationRowLookupMechanisms};
use eredu_checkpoint::recipe::{RecipeDtype, RecipeMetadata};
use eredu_core::residency::{OffloadConfig, OffloadUnitId, OffloadUnitRange, ResidencyPolicy};
use eredu_nn::{mechanism_memory::*, ParameterId, TensorElementType};
use eredu_runtime::{
    AddressableStorageCapabilities, ParameterBankLoadOptions, RowEncoding, RowLookupDescriptor,
    RowLookupDescriptors, RowLookupError, RowLookupLimits, RowLookupSpec, RowLookupWorkspace,
    SelectedRowLookupPlans,
};
use std::cell::RefCell;

#[derive(Default)]
struct Provider {
    floating_dtype: Option<eredu_runtime::StateStorageDtype>,
    no_storage: bool,
    no_decoder: bool,
    workspace_extra: u64,
    completion_retention: bool,
    workspace_error: bool,
    memory_error: bool,
}

impl PreparationMechanismProvider for Provider {
    fn floating_state_dtype(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<eredu_runtime::StateStorageDtype> {
        assert_eq!(source, &eredu_core::checkpoint::TensorDtype::U32);
        self.floating_dtype
    }
    fn row_lookup_storage(&self) -> Option<AddressableStorageCapabilities> {
        (!self.no_storage).then(|| AddressableStorageCapabilities::new(true, true, true, 4096))
    }
    fn row_lookup_workspace(
        &self,
        descriptor: &RowLookupDescriptor,
    ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
        assert_eq!(descriptor, &rows());
        if self.workspace_error {
            return Err(RowLookupError::Geometry);
        }
        Ok((!self.no_decoder).then_some(RowLookupWorkspace {
            decode_bytes: 80 + self.workspace_extra,
            scalar_bytes: 0,
        }))
    }
    fn row_lookup_decode_memory(
        &self,
        descriptor: &RowLookupDescriptor,
    ) -> Result<MechanismMemoryContract, RowLookupError> {
        assert_eq!(descriptor, &rows());
        if self.memory_error {
            return Err(RowLookupError::Geometry);
        }
        let mut facts = descriptor.decode_memory()?;
        facts.storage.push(MechanismStorage {
            name: "decoded output".into(),
            role: MechanismStorageRole::Output,
            payload: MechanismBytes::exact(16),
            capacity: MechanismBytes::exact(16),
            backing: MechanismBacking::Invocation,
            placement: MechanismPlacement::Execution,
            retention: if self.completion_retention {
                StorageRetention::NativeCompletion
            } else {
                StorageRetention::Returned
            },
            detail: "neutral output backing".into(),
        });
        Ok(facts)
    }
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        panic!("row selection must only query row facts")
    }
    fn supports_grouped_operation(&self, _: eredu_runtime::GroupedOperationRequirement) -> bool {
        panic!("row selection must only query row facts")
    }
    fn replicated_text_capabilities(
        &self,
        _: &ReplicatedTextRequirements,
        _: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        panic!("row selection must only query row facts")
    }
    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        panic!("row selection must only query row facts")
    }
    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        panic!("row selection must only query row facts")
    }
    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        panic!("row selection must only query row facts")
    }
}

fn rows() -> RowLookupDescriptor {
    RowLookupDescriptor::new(
        OffloadUnitRange::new(
            OffloadUnitId::new("fixture.rows").unwrap(),
            0,
            4,
            8,
            ResidencyPolicy::Cacheable,
        )
        .unwrap(),
        RecipeMetadata {
            shape: vec![4, 2],
            dtype: RecipeDtype::F32,
            byte_len: 32,
        },
        RowLookupSpec {
            parameter: ParameterId::new("table").unwrap(),
            bank: 1,
            unit: 0,
            rows: 4,
            dimensions: 2,
            encoding: RowEncoding::Dense,
            output_type: TensorElementType::F32,
        },
        None,
        RowLookupLimits {
            requests: 8,
            rows_per_acquisition: 2,
            acquisition_bytes: 16,
            host_bytes: 1024,
            output_bytes: 192,
        },
    )
    .unwrap()
}

fn select(
    provider: &impl PreparationMechanismProvider,
) -> Result<SelectedRowLookupPlans, eredu_runtime::RowLookupSelectionError> {
    SelectedRowLookupPlans::select(
        RowLookupDescriptors::new([rows()], 1).unwrap(),
        ParameterBankLoadOptions::new(
            OffloadConfig::new(Some(16), Some(0), 1).unwrap(),
            4096,
            4096,
        )
        .unwrap(),
        0,
        &PreparationRowLookupMechanisms::new(provider),
    )
}

#[test]
fn exact_row_facts_are_recorded_and_changed_mechanisms_invalidate_agreement() {
    let provider = Provider::default();
    let recorder = RecordingMechanisms {
        provider: &provider,
        observations: RefCell::new(vec![]),
    };
    let selected = select(&recorder).unwrap();
    assert_eq!(selected.requirements().decode_bytes, 80);
    let observations = recorder.observations.into_inner();
    assert_eq!(observations.len(), 3);
    assert!(observations.iter().all(|fact| fact.matches(&provider)));
    for changed in [
        Provider {
            no_storage: true,
            ..Default::default()
        },
        Provider {
            no_decoder: true,
            ..Default::default()
        },
        Provider {
            workspace_extra: 8,
            ..Default::default()
        },
        Provider {
            completion_retention: true,
            ..Default::default()
        },
    ] {
        assert!(!observations.iter().all(|fact| fact.matches(&changed)));
    }
}

#[test]
fn unsupported_rows_are_retained_until_the_exact_support_changes() {
    for provider in [
        Provider {
            no_storage: true,
            ..Default::default()
        },
        Provider {
            no_decoder: true,
            ..Default::default()
        },
    ] {
        let recorder = RecordingMechanisms {
            provider: &provider,
            observations: RefCell::new(vec![]),
        };
        assert!(select(&recorder).is_err());
        let observations = recorder.observations.into_inner();
        assert!(observations.iter().all(|fact| fact.matches(&provider)));
        assert!(!observations
            .iter()
            .all(|fact| fact.matches(&Provider::default())));
    }
}

#[test]
fn failed_row_probes_never_reuse_cached_selection_authority() {
    for provider in [
        Provider {
            workspace_error: true,
            ..Default::default()
        },
        Provider {
            memory_error: true,
            ..Default::default()
        },
    ] {
        let recorder = RecordingMechanisms {
            provider: &provider,
            observations: RefCell::new(vec![]),
        };
        assert!(select(&recorder).is_err());
        let observations = recorder.observations.into_inner();
        assert!(!observations.iter().all(|fact| fact.matches(&provider)));
        assert!(!observations
            .iter()
            .all(|fact| fact.matches(&Provider::default())));
    }
}

#[test]
fn embedding_materialization_dtype_changes_invalidate_cached_authority() {
    use eredu_runtime::StateStorageDtype;
    for admitted in [
        None,
        Some(StateStorageDtype::F32),
        Some(StateStorageDtype::Bf16),
    ] {
        let provider = Provider {
            floating_dtype: admitted,
            ..Default::default()
        };
        let recorder = RecordingMechanisms {
            provider: &provider,
            observations: RefCell::new(vec![]),
        };
        assert_eq!(
            recorder.floating_state_dtype(&eredu_core::checkpoint::TensorDtype::U32),
            admitted
        );
        let observations = recorder.observations.into_inner();
        assert_eq!(observations.len(), 1);
        for current in [
            None,
            Some(StateStorageDtype::F32),
            Some(StateStorageDtype::Bf16),
        ] {
            let changed = Provider {
                floating_dtype: current,
                ..Default::default()
            };
            assert_eq!(observations[0].matches(&changed), admitted == current);
        }
    }
}
