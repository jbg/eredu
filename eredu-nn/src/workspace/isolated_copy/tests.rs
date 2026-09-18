use super::*;
use crate::Tensor;

mod finite;

#[derive(Clone, Copy, Debug, Default)]
enum DeepEffect {
    #[default]
    Independent,
    Alias,
    PossibleAlias,
}

#[derive(Clone, Debug)]
struct Facts {
    calls: Rc<Cell<usize>>,
    deep: DeepEffect,
    missing_tensor: Option<usize>,
    missing_host: Option<usize>,
    scratch: u64,
    host: u64,
    output_bytes: Option<u64>,
}

impl Default for Facts {
    fn default() -> Self {
        Self {
            calls: Rc::new(Cell::new(0)),
            deep: DeepEffect::Independent,
            missing_tensor: None,
            missing_host: None,
            scratch: 5,
            host: 3,
            output_bytes: None,
        }
    }
}

impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let index = self.calls.get();
        self.calls.set(index + 1);
        if self.missing_tensor == Some(index) {
            return Ok(None);
        }
        let bytes = self.output_bytes.unwrap_or(operation.outputs[0].bytes()?);
        let effect = match operation.kind {
            WorkspaceOperationKind::Contiguous => WorkspaceOutputStorage::AllocateOrAliasInputs {
                bytes,
                inputs: vec![0],
            },
            WorkspaceOperationKind::View(_) | WorkspaceOperationKind::Transpose(_) => WorkspaceOutputStorage::AliasInput(0),
            WorkspaceOperationKind::DeepCopy => match self.deep {
                DeepEffect::Independent => WorkspaceOutputStorage::Allocate(bytes),
                DeepEffect::Alias => WorkspaceOutputStorage::AliasInput(0),
                DeepEffect::PossibleAlias => WorkspaceOutputStorage::AllocateOrAliasInputs {
                    bytes,
                    inputs: vec![0],
                },
            },
            _ => WorkspaceOutputStorage::Allocate(bytes),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![effect],
            scratch_bytes: self.scratch,
            assumptions: "test: exact per-operation buffers and retained scratch".into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(
            (self.missing_host != Some(self.calls.get() - 1)).then(|| WorkspaceHostBound {
                bytes: self.host,
                assumptions: "test: disjoint operation-owned staging".into(),
            }),
        )
    }
}

fn imported(
    context: &WorkspaceContext,
    root: &WorkspaceExistingStorage,
    shape: &[i32],
) -> WorkspaceTensor {
    WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
        root,
        context,
    )
    .unwrap()
}

#[test]
fn aliases_pin_one_source_but_copy_every_slot_and_keep_prior_destinations() {
    let facts = Facts::default();
    let calls = facts.calls.clone();
    let context = WorkspaceContext::new(facts);
    let root = WorkspaceExistingStorage::new(Some(128), &context);
    let borrowed = WorkspaceBorrowedStorage::new(&context, [&root, &root]).unwrap();
    let sources = [
        imported(&context, &root, &[2, 2]),
        imported(&context, &root, &[8]),
    ];
    let plan = WorkspaceIsolatedCopyPlan::prepare(&context, &borrowed, &sources).unwrap();
    assert_eq!(calls.get(), 4);
    assert!(plan.source_storage().same_identity(&borrowed));
    assert_eq!(plan.source_storage().roots().len(), 1);
    assert_eq!(
        plan.source_layouts(),
        &[sources[0].layout().clone(), sources[1].layout().clone()]
    );
    let report = plan.report();
    assert_eq!(report.operations.len(), 4);
    for pair in report.operations.chunks_exact(2) {
        assert!(matches!(pair[0].kind, WorkspaceOperationKind::Contiguous));
        assert!(matches!(pair[1].kind, WorkspaceOperationKind::DeepCopy));
    }
    // Two contiguous buffers (16 + 32), two destinations (16 + 32),
    // four scratch bounds (5 each), and four host bounds (3 each).
    assert_eq!(plan.incremental_bytes(), Some(128));
    assert_eq!(report.total_bytes, Some(128));
    assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(176));
    let residual = report.residual.as_ref().unwrap();
    assert_eq!(residual.retained_bytes, Some(48));
    assert_eq!(residual.transient_bytes, Some(80));
    assert!(!residual.borrowed_storage.same_identity(&borrowed));
    assert!(context.report(&[]).unwrap().operations.is_empty());
    assert!(context.report(&[]).unwrap().residual.is_none());
}

#[test]
fn invalid_sources_reject_before_invoking_selected_facts() {
    let facts = Facts::default();
    let calls = facts.calls.clone();
    let context = WorkspaceContext::new(facts);
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let selected = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
    let source = imported(&context, &root, &[4]);
    let derived_allocate = source.square(&context).unwrap();
    let derived_alias = source.reshape(&[2, 2], &context).unwrap();
    assert!(Rc::ptr_eq(&derived_alias.storage, &source.storage));
    let implicit = WorkspaceTensor::existing(source.layout().clone(), &context).unwrap();
    let before = calls.get();
    for invalid in [derived_allocate, derived_alias, implicit] {
        assert!(matches!(
            WorkspaceIsolatedCopyPlan::prepare(&context, &selected, &[invalid]),
            Err(WorkspaceCopyError::SourceNotImported { index: 0 })
        ));
    }
    let unselected = WorkspaceExistingStorage::new(Some(64), &context);
    assert!(matches!(
        WorkspaceIsolatedCopyPlan::prepare(
            &context,
            &selected,
            &[imported(&context, &unselected, &[4])]
        ),
        Err(WorkspaceCopyError::SourceStorageMismatch { index: 0 })
    ));
    let unknown = WorkspaceExistingStorage::new(None, &context);
    assert!(matches!(
        WorkspaceIsolatedCopyPlan::prepare(
            &context,
            &selected,
            &[imported(&context, &unknown, &[4])]
        ),
        Err(WorkspaceCopyError::UnknownSourceCapacity { index: 0 })
    ));
    let other = WorkspaceContext::new(Facts::default());
    let foreign = WorkspaceExistingStorage::new(Some(64), &other);
    let foreign_selection = WorkspaceBorrowedStorage::new(&other, [&foreign]).unwrap();
    assert!(matches!(
        WorkspaceIsolatedCopyPlan::prepare(&context, &foreign_selection, &[source.clone()]),
        Err(WorkspaceCopyError::ContextMismatch)
    ));
    assert!(matches!(
        WorkspaceIsolatedCopyPlan::prepare(
            &context,
            &selected,
            &[imported(&other, &foreign, &[4])]
        ),
        Err(WorkspaceCopyError::ContextMismatch)
    ));
    assert_eq!(calls.get(), before);
}

#[test]
fn selection_must_match_exactly_including_zero_byte_identities_and_empty_plans() {
    let context = WorkspaceContext::new(Facts {
        scratch: 0,
        host: 0,
        ..Facts::default()
    });
    let first = WorkspaceExistingStorage::new(Some(16), &context);
    let second = WorkspaceExistingStorage::new(Some(16), &context);
    let selected = WorkspaceBorrowedStorage::new(&context, [&first, &second]).unwrap();
    assert!(matches!(
        WorkspaceIsolatedCopyPlan::prepare(
            &context,
            &selected,
            &[imported(&context, &first, &[4])]
        ),
        Err(WorkspaceCopyError::SourceSetMismatch)
    ));
    assert!(matches!(
        WorkspaceIsolatedCopyPlan::prepare(&context, &selected, &[]),
        Err(WorkspaceCopyError::SourceSetMismatch)
    ));
    let empty = WorkspaceBorrowedStorage::new(&context, []).unwrap();
    let plan = WorkspaceIsolatedCopyPlan::prepare(&context, &empty, &[]).unwrap();
    assert_eq!(plan.incremental_bytes(), Some(0));
    assert!(plan.report().operations.is_empty());
    let zero = WorkspaceExistingStorage::new(Some(0), &context);
    let zero_selection = WorkspaceBorrowedStorage::new(&context, [&zero]).unwrap();
    let zero_source = imported(&context, &zero, &[0]);
    assert!(matches!(
        WorkspaceIsolatedCopyPlan::prepare(&context, &empty, &[zero_source.clone()]),
        Err(WorkspaceCopyError::SourceStorageMismatch { index: 0 })
    ));
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, &zero_selection, &[zero_source]).unwrap();
    assert_eq!(plan.incremental_bytes(), Some(0));
    assert_eq!(plan.report().operations.len(), 2);
    assert_eq!(plan.source_storage().roots().len(), 1);
}

#[test]
fn explicit_imported_broadcast_view_is_valid_even_when_larger_than_source_capacity() {
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(4), &context);
    let borrowed = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
    let view = imported(&context, &root, &[3, 4]);
    let plan = WorkspaceIsolatedCopyPlan::prepare(&context, &borrowed, &[view]).unwrap();
    assert_eq!(plan.incremental_bytes(), Some(112));
    assert_eq!(
        plan.report().state.as_ref().unwrap().retained_bytes,
        Some(52)
    );
}

#[test]
fn known_deep_copy_alias_effects_cannot_authorize_independent_destinations() {
    for deep in [DeepEffect::Alias, DeepEffect::PossibleAlias] {
        let context = WorkspaceContext::new(Facts {
            deep,
            ..Facts::default()
        });
        let root = WorkspaceExistingStorage::new(Some(64), &context);
        let borrowed = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
        assert!(matches!(
            WorkspaceIsolatedCopyPlan::prepare(
                &context,
                &borrowed,
                &[imported(&context, &root, &[4])]
            ),
            Err(WorkspaceCopyError::NonIndependentDestination { index: 0 })
        ));
    }
}

#[test]
fn missing_tensor_or_host_fact_is_sticky_across_later_priced_copies() {
    for missing_tensor in [true, false] {
        for index in [0, 1] {
            let context = WorkspaceContext::new(Facts {
                missing_tensor: missing_tensor.then_some(index),
                missing_host: (!missing_tensor).then_some(index),
                ..Facts::default()
            });
            let root = WorkspaceExistingStorage::new(Some(64), &context);
            let selected = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
            let source = imported(&context, &root, &[4]);
            let plan =
                WorkspaceIsolatedCopyPlan::prepare(&context, &selected, &[source.clone(), source])
                    .unwrap();
            assert_eq!(plan.incremental_bytes(), None);
            assert_eq!(plan.report().total_bytes, None);
            assert_eq!(plan.report().operations.len(), 4);
            if missing_tensor {
                assert_eq!(plan.report().unpriced_operations, [index]);
            } else {
                assert_eq!(plan.report().unpriced_host_operations, [index]);
            }
        }
    }
}

#[test]
fn full_union_scratch_host_and_incremental_overflow_remain_typed_errors() {
    for (source_bytes, facts) in [
        (u64::MAX, Facts::default()),
        (
            4,
            Facts {
                scratch: u64::MAX,
                ..Facts::default()
            },
        ),
        (
            4,
            Facts {
                host: u64::MAX,
                ..Facts::default()
            },
        ),
        (
            4,
            Facts {
                output_bytes: Some(u64::MAX),
                ..Facts::default()
            },
        ),
    ] {
        let context = WorkspaceContext::new(facts);
        let root = WorkspaceExistingStorage::new(Some(source_bytes), &context);
        let borrowed = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
        assert!(matches!(
            WorkspaceIsolatedCopyPlan::prepare(
                &context,
                &borrowed,
                &[imported(&context, &root, &[1])]
            ),
            Err(WorkspaceCopyError::Overflow { .. })
        ));
        assert!(context.report(&[]).unwrap().operations.is_empty());
    }
}

struct ResetOriginal {
    original: Rc<RefCell<Option<WorkspaceContext>>>,
    facts: Facts,
}
impl fmt::Debug for ResetOriginal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResetOriginal").finish_non_exhaustive()
    }
}
impl WorkspaceMechanisms for ResetOriginal {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if let Some(original) = self.original.borrow().as_ref() {
            original.begin_span();
        }
        self.facts.operation_bound(operation)
    }
    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        self.facts.host_workspace_bound(operation)
    }
}

#[test]
fn callback_and_caller_resets_cannot_mutate_the_sealed_private_trace() {
    let original = Rc::new(RefCell::new(None));
    let context = WorkspaceContext::new(ResetOriginal {
        original: original.clone(),
        facts: Facts::default(),
    });
    original.replace(Some(context.clone()));
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let borrowed = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
    let source = imported(&context, &root, &[4]);
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, &borrowed, &[source.clone(), source.clone()])
            .unwrap();
    assert_eq!(plan.incremental_bytes(), Some(96));
    assert_eq!(plan.report().operations.len(), 4);
    let mut editable_diagnostic = plan.report().clone();
    editable_diagnostic.total_bytes = Some(0);
    editable_diagnostic.operations.clear();
    editable_diagnostic.residual.as_mut().unwrap().total_bytes = Some(0);
    context.begin_span();
    source.square(&context).unwrap();
    context.begin_state_span([&source]).unwrap();
    assert_eq!(plan.incremental_bytes(), Some(96));
    assert_eq!(plan.report().operations.len(), 4);
    assert_eq!(
        plan.report().residual.as_ref().unwrap().total_bytes,
        Some(96)
    );
    assert!(plan.source_storage().same_identity(&borrowed));
    // Break the deliberately retained callback/context cycle in this fixture.
    original.take();
}

#[derive(Debug, thiserror::Error)]
#[error("selected copy mechanism failed")]
struct SelectedFailure;

#[derive(Debug)]
struct FailingFacts(Rc<Cell<usize>>);
impl WorkspaceMechanisms for FailingFacts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        self.0.set(self.0.get() + 1);
        Err(Error::backend_retained_source(SelectedFailure))
    }
}

#[test]
fn selected_mechanism_failure_preserves_its_source_and_stops_before_deep_copy() {
    use std::error::Error as _;
    let calls = Rc::new(Cell::new(0));
    let context = WorkspaceContext::new(FailingFacts(calls.clone()));
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let borrowed = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
    let error =
        WorkspaceIsolatedCopyPlan::prepare(&context, &borrowed, &[imported(&context, &root, &[4])])
            .unwrap_err();
    let WorkspaceCopyError::Mechanism(error) = error else {
        panic!("original mechanism failure must survive")
    };
    assert!(error.source().unwrap().is::<SelectedFailure>());
    assert_eq!(calls.get(), 1);
    assert!(context.report(&[]).unwrap().operations.is_empty());
}
