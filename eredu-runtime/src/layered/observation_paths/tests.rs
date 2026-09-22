use super::*;
use std::{
    convert::Infallible,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

pub(super) struct PayloadRetired(Arc<AtomicBool>);
impl Drop for PayloadRetired {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
fn spare(text: &str, capacity: usize) -> String {
    let mut value = String::with_capacity(capacity);
    value.push_str(text);
    value
}
fn source() -> SharedLayeredObservationPaths {
    SharedLayeredObservationPaths(Arc::new(PathsInner {
        media_prefill: Vec::new().into_boxed_slice(),
        payload: vec![GroupPaths {
            id: spare("decoder", 37),
            units: vec![
                UnitPaths {
                    input: spare("layer.0.input", 67),
                    output: spare("layer.0.output", 89),
                    effective_input: spare("layer.0.input.effective", 97),
                    effective_output: spare("layer.0.output.effective", 101),
                    outer: true,
                },
                UnitPaths {
                    input: spare("layer.1.input", 109),
                    output: spare("layer.1.output", 131),
                    effective_input: spare("layer.1.input.effective", 137),
                    effective_output: spare("layer.1.output.effective", 139),
                    outer: false,
                },
            ]
            .into_boxed_slice(),
            input: Some(spare("merge.output", 151)),
            output: Some(spare("group.output", 173)),
        }]
        .into_boxed_slice(),
        prefill: Box::new([]),
        retired: None,
        custody: MetadataCustody::new(),
    }))
}

#[test]
fn capacity_measures_actual_spare_strings_and_boxed_entries() {
    let owner = source();
    let group = &owner.0.payload[0];
    let expected = 2 * size_of::<Box<[PrefillObservationDeclaration]>>()
        + size_of::<Box<[GroupPaths]>>()
        + size_of::<GroupPaths>()
        + 2 * size_of::<UnitPaths>()
        + group.id.capacity()
        + group.input.as_ref().unwrap().capacity()
        + group.output.as_ref().unwrap().capacity()
        + group
            .units
            .iter()
            .map(|unit| {
                unit.input.capacity()
                    + unit.output.capacity()
                    + unit.effective_input.capacity()
                    + unit.effective_output.capacity()
            })
            .sum::<usize>();
    assert_eq!(owner.capacity_bytes(), Some(expected as u64));
    assert_eq!(owner.group_count(), 1);
    assert_eq!(owner.unit_count(0), Some(2));
    assert_eq!(owner.unit_count(1), None);
    assert_eq!(
        owner.unit_paths(0, 1),
        Some(("layer.1.input", "layer.1.output"))
    );
    assert_eq!(owner.outer_unit_paths(0, 1), None);
    assert_eq!(owner.unit_paths(0, 2), None);
    assert_eq!(extent::<u64>(usize::MAX), None);
    assert_eq!(extent::<()>(usize::MAX), Some(0));
    let empty = SharedLayeredObservationPaths(Arc::new(PathsInner {
        media_prefill: Vec::new().into_boxed_slice(),
        payload: Box::new([]),
        prefill: Box::new([]),
        retired: None,
        custody: MetadataCustody::new(),
    }));
    assert_eq!(
        empty.capacity_bytes(),
        Some(
            (size_of::<Box<[GroupPaths]>>() + 2 * size_of::<Box<[PrefillObservationDeclaration]>>())
                as u64
        )
    );
}

struct Charge {
    retired: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
    other: SharedLayeredObservationPaths,
}
impl Drop for Charge {
    fn drop(&mut self) {
        assert!(self.retired.load(Ordering::SeqCst));
        self.other
            .try_attach(&SharedStorageAccountingId::default(), || {
                Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(()))
            })
            .unwrap();
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn aliases_share_physical_strings_and_all_domains_retire_after_payload() {
    let retired = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut owner = source();
    Arc::get_mut(&mut owner.0).unwrap().retired = Some(PayloadRetired(retired.clone()));
    let alias = owner.clone();
    let identity = owner.identity().clone();
    let independent = source();
    assert_ne!(identity, *independent.identity());
    assert!(owner.same_storage(&alias));
    assert!(!owner.same_storage(&independent));
    assert_eq!(
        owner.unit_paths(0, 0).unwrap().0.as_ptr(),
        alias.unit_paths(0, 0).unwrap().0.as_ptr()
    );
    let domain = SharedStorageAccountingId::default();
    for domain in [&domain, &SharedStorageAccountingId::default()] {
        assert!(owner
            .try_attach(domain, || Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(
                Charge {
                    retired: retired.clone(),
                    drops: drops.clone(),
                    other: independent.clone()
                }
            )))
            .unwrap());
    }
    assert!(!alias
        .try_attach::<Infallible>(&domain, || panic!("duplicate provider"))
        .unwrap());
    let inventory = crate::SharedHostMetadata::ObservationPaths(alias);
    assert_eq!(inventory.identity(), &identity);
    assert_eq!(inventory.capacity_bytes(), owner.capacity_bytes());
    drop(owner);
    assert!(!retired.load(Ordering::SeqCst));
    drop(inventory);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    // A surviving payload-free key does not keep either payload or charges alive.
    drop(identity);
}

#[derive(Debug, thiserror::Error)]
#[error("attachment sentinel")]
struct AttachmentFailure;
#[test]
fn provider_failure_is_retryable_and_poison_preserves_earlier_custody() {
    let mut owner = source();
    let retired = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    Arc::get_mut(&mut owner.0).unwrap().retired = Some(PayloadRetired(retired.clone()));
    let domain = SharedStorageAccountingId::default();
    assert!(matches!(
        owner.try_attach(&domain, || Err(AttachmentFailure)),
        Err(SharedStorageAttachmentError::Provider(AttachmentFailure))
    ));
    assert!(owner
        .try_attach(&domain, || Ok::<Box<dyn Send + Sync>, Infallible>(
            Box::new(Charge {
                retired: retired.clone(),
                drops: drops.clone(),
                other: source()
            })
        ))
        .unwrap());
    let alias = owner.clone();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = owner.try_attach::<Infallible>(&SharedStorageAccountingId::default(), || {
            panic!("provider unwind")
        });
    }))
    .is_err());
    assert!(matches!(
        owner.try_attach::<Infallible>(&SharedStorageAccountingId::default(), || panic!(
            "poison cannot acquire"
        )),
        Err(SharedStorageAttachmentError::Poisoned)
    ));
    drop(owner);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn runtime_stamp_invalidation_never_mutates_the_shared_source() {
    let source = source();
    let mut first = ObservationBinding::new();
    let second = ObservationBinding::new();
    let prepared = first
        .prepare(source.clone(), &metadata::Destination::<Infallible>(None))
        .unwrap();
    let rebound = second
        .prepare(source.clone(), &metadata::Destination::<Infallible>(None))
        .unwrap();
    first.validate::<Infallible>(&prepared).unwrap();
    assert!(matches!(
        second.validate::<Infallible>(&prepared),
        Err(PreparedLayeredObservationError::BindingMismatch)
    ));
    assert!(rebound.source().same_storage(prepared.source()));
    assert!(prepared.traversal_host_peak_bytes().unwrap() > 0);
    first.invalidate();
    assert!(matches!(
        first.validate::<Infallible>(&prepared),
        Err(PreparedLayeredObservationError::BindingMismatch)
    ));
    let fresh = first
        .prepare(source.clone(), &metadata::Destination::<Infallible>(None))
        .unwrap();
    first.validate::<Infallible>(&fresh).unwrap();
    second.validate::<Infallible>(&rebound).unwrap();
    assert!(fresh.source().same_storage(prepared.source()));
}

#[test]
fn generated_selection_traces_real_operations_in_the_original_context_and_preserves_unknown() {
    use eredu_nn::{
        workspace::{
            WorkspaceBackend, WorkspaceContext, WorkspaceDtype, WorkspaceHostBound,
            WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
            WorkspaceOutputStorage, WorkspaceTensor,
        },
        Error, Tensor,
    };
    #[derive(Debug)]
    struct Facts(bool);
    impl WorkspaceMechanisms for Facts {
        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            if !self.0 {
                return Ok(None);
            }
            Ok(Some(WorkspaceOperationBound {
                outputs: operation
                    .outputs
                    .iter()
                    .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                    .collect::<Result<_, _>>()?,
                scratch_bytes: 0,
                assumptions: "fixture exact output allocation".into(),
            }))
        }
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "fixture host mechanism".into(),
            }))
        }
    }
    struct Observer {
        selected: bool,
        values: Vec<WorkspaceTensor>,
    }
    impl ActivationObserver<WorkspaceTensor, Error> for Observer {
        fn observe(&mut self, _: &str, value: &WorkspaceTensor) -> Result<(), Error> {
            self.values.push(value.clone());
            Ok(())
        }
        fn observe_generated(
            &mut self,
            path: &str,
            _: &WorkspaceTensor,
            _: &eredu_core::capture::GeneratedCaptureSource,
            generate: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
        ) -> Result<(), Error> {
            if self.selected {
                self.observe(path, &generate()?)?;
            }
            Ok(())
        }
    }
    for known in [false, true] {
        for selected in [false, true] {
            let context = WorkspaceContext::new(Facts(known));
            let prototype = WorkspaceTensor::existing(
                WorkspaceLayout::new(&[1, 4], WorkspaceDtype::Float32).unwrap(),
                &context,
            )
            .unwrap();
            let paths = source();
            let calls = std::cell::Cell::new(0);
            let mut observer = Observer {
                selected,
                values: Vec::new(),
            };
            let mut hook = BorrowedHook {
                observer: &mut observer,
                paths: &paths,
            };
            <BorrowedHook<'_,Observer> as LayeredTraversalHook<WorkspaceBackend,(),Error>>::observe_generated_activation(
            &mut hook,"generated",&prototype,&eredu_core::capture::GeneratedCaptureSource {creation_bytes:16,source_dtype:Some(eredu_core::checkpoint::TensorDtype::F32)},
            &mut || { calls.set(calls.get()+1); prototype.tanh(&context) },
        ).unwrap();
            assert_eq!(calls.get(), usize::from(selected));
            let report = context.report(&observer.values).unwrap();
            assert_eq!(
                report.total_bytes,
                if !selected {
                    Some(0)
                } else if known {
                    Some(16)
                } else {
                    None
                }
            );
            for value in &observer.values {
                assert!(value.same_context(&prototype));
            }
        }
    }
}

mod retained_factory;

mod prefill;

mod fragments;

#[test]
fn original_fingerprint_rejects_foreign_runtime_and_same_source_rebinding() {
    let source = source();
    let mut original = ObservationBinding::new();
    let token = original
        .prepare(source.clone(), &metadata::Destination::<Infallible>(None))
        .unwrap();
    let fingerprint = token.binding_identity();
    assert!(fingerprint.matches(&token));
    for _ in 0..4 {
        let same = original
            .prepare(source.clone(), &metadata::Destination::<Infallible>(None))
            .unwrap();
        original.validate::<Infallible>(&same).unwrap();
        assert!(fingerprint.matches(&same));
    }
    let foreign = ObservationBinding::new();
    let foreign_token = foreign
        .prepare(source.clone(), &metadata::Destination::<Infallible>(None))
        .unwrap();
    foreign.validate::<Infallible>(&foreign_token).unwrap();
    assert!(foreign_token.source().same_storage(token.source()));
    assert!(!fingerprint.matches(&foreign_token));
    original.invalidate();
    assert!(original.validate::<Infallible>(&token).is_err());
    let rebound = original
        .prepare(source, &metadata::Destination::<Infallible>(None))
        .unwrap();
    original.validate::<Infallible>(&rebound).unwrap();
    assert!(!fingerprint.matches(&rebound));
}

#[test]
fn fingerprint_keeps_only_original_control_block_not_prepared_runtime_alive() {
    let source = source();
    let mut original = ObservationBinding::new();
    let token = original
        .prepare(source.clone(), &metadata::Destination::<Infallible>(None))
        .unwrap();
    let fingerprint = token.binding_identity();
    assert_eq!(fingerprint.runtime.strong_count(), 2);
    original.invalidate();
    drop(token);
    assert_eq!(fingerprint.runtime.strong_count(), 0);
    let rebound = original
        .prepare(source, &metadata::Destination::<Infallible>(None))
        .unwrap();
    assert!(!fingerprint.matches(&rebound));
}

mod media_capture;
