use super::*;
use std::convert::Infallible;

const TENSOR: &str = "test: exact per-operation buffers and retained scratch";
const HOST: &str = "test: disjoint operation-owned staging";

#[derive(Default)]
struct FiniteFacts {
    deep: DeepEffect,
    missing_deep: bool,
}

impl WorkspaceFactMechanisms for FiniteFacts {
    type Error = Infallible;

    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        if self.missing_deep && matches!(operation.kind, WorkspaceOperationKindView::DeepCopy) {
            return Ok(None);
        }
        let aliases = usize::from(
            matches!(operation.kind, WorkspaceOperationKindView::Contiguous)
                || matches!(self.deep, DeepEffect::PossibleAlias),
        );
        Ok(Some(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: 1,
                aliases,
                assumption_bytes: TENSOR.len(),
            },
            scratch_bytes: 5,
        }))
    }

    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        let Some(facts) = self.operation_facts(operation)? else {
            return Ok(None);
        };
        destination.validate(facts.layout).unwrap();
        let bytes = operation.outputs.get(0).unwrap().bytes().unwrap();
        destination.outputs[0] = if facts.layout.aliases == 1 {
            destination.aliases[0] = 0;
            WorkspaceOutputEffect::AllocateOrAliasInputs {
                bytes,
                alias_start: 0,
                alias_count: 1,
            }
        } else if matches!(self.deep, DeepEffect::Alias) {
            WorkspaceOutputEffect::AliasInput(0)
        } else {
            WorkspaceOutputEffect::Allocate(bytes)
        };
        destination.assumptions.copy_from_slice(TENSOR.as_bytes());
        Ok(Some(facts))
    }

    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(Some(WorkspaceHostFacts {
            bytes: 3,
            assumption_bytes: HOST.len(),
        }))
    }

    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        let facts = self.host_facts(operation)?.unwrap();
        destination.validate(facts).unwrap();
        destination.assumptions.copy_from_slice(HOST.as_bytes());
        Ok(Some(facts))
    }
}

#[test]
fn finite_copy_matches_ordinary_and_rejects_changed_layout_or_aliasing() {
    let facts = FiniteFacts::default();
    // Quote while only borrowed source shapes exist, before context/imports.
    let shapes: [&[i32]; 2] = [&[2, 2], &[8]];
    let mut builder = WorkspaceCopyPreparationLayoutBuilder::new();
    for shape in shapes {
        builder
            .push_source(
                WorkspaceLayoutView::new(shape, WorkspaceDtype::Float32).unwrap(),
                &facts,
            )
            .unwrap();
    }
    let layout = builder.finish(2).unwrap();
    let ordinary_facts = Facts::default();
    let ordinary_calls = ordinary_facts.calls.clone();
    let context = WorkspaceContext::new(ordinary_facts);
    let root = WorkspaceExistingStorage::new(Some(128), &context);
    let borrowed = WorkspaceBorrowedStorage::new_finite(&context, [&root, &root], 2).unwrap();
    let sources = shapes.map(|shape| imported(&context, &root, shape));
    let finite = WorkspaceIsolatedCopyPlan::prepare_finite_with_layout(
        &context, &borrowed, &sources, &facts, layout,
    )
    .unwrap()
    .construct()
    .unwrap();
    assert_eq!(ordinary_calls.get(), 0, "finite facts must not fall back");
    let ordinary = WorkspaceIsolatedCopyPlan::prepare(&context, &borrowed, &sources).unwrap();
    assert_eq!(finite.incremental_bytes(), Some(128));
    assert_eq!(finite.incremental_bytes(), ordinary.incremental_bytes());
    assert_eq!(finite.source_layouts(), ordinary.source_layouts());
    assert_eq!(finite.report().operations.len(), 4);
    for (actual, expected) in finite
        .report()
        .operations
        .iter()
        .zip(&ordinary.report().operations)
    {
        assert!(matches!(
            (&actual.kind, &expected.kind),
            (
                WorkspaceOperationKind::Contiguous,
                WorkspaceOperationKind::Contiguous
            ) | (
                WorkspaceOperationKind::DeepCopy,
                WorkspaceOperationKind::DeepCopy
            )
        ));
        assert_eq!(actual.inputs, expected.inputs);
        assert_eq!(actual.outputs, expected.outputs);
    }
    assert_eq!(finite.report().assumptions, ordinary.report().assumptions);
    assert_eq!(
        finite.report().state.as_ref().unwrap().retained_bytes,
        Some(176)
    );
    let residual = finite.report().residual.as_ref().unwrap();
    assert_eq!(residual.retained_bytes, Some(48));
    assert_eq!(residual.transient_bytes, Some(80));
    assert!(residual.borrowed_storage.same_identity(&borrowed));

    // A shape/population change invalidates the accepted quote before filling.
    let changed = [imported(&context, &root, &[4]), sources[1].clone()];
    assert!(matches!(
        WorkspaceIsolatedCopyPlan::prepare_finite_with_layout(
            &context, &borrowed, &changed, &facts, layout,
        ),
        Err(WorkspaceCopyPreparationError::Facts)
    ));
    for deep in [DeepEffect::Alias, DeepEffect::PossibleAlias] {
        let invalid = FiniteFacts {
            deep,
            missing_deep: false,
        };
        assert!(matches!(
            WorkspaceIsolatedCopyPlan::prepare_finite(&context, &borrowed, &sources, &invalid)
                .unwrap()
                .construct(),
            Err(WorkspaceCopyPreparationError::Source(
                WorkspaceCopyError::NonIndependentDestination { index: 0 }
            ))
        ));
    }
    let unknown = FiniteFacts {
        missing_deep: true,
        ..FiniteFacts::default()
    };
    let unknown =
        WorkspaceIsolatedCopyPlan::prepare_finite(&context, &borrowed, &sources, &unknown)
            .unwrap()
            .construct()
            .unwrap();
    assert_eq!(unknown.incremental_bytes(), None);
    assert_eq!(unknown.report().unpriced_operations, vec![1, 3]);
    assert_eq!(
        ordinary_calls.get(),
        4,
        "no ordinary fallback on refusal or unknown"
    );
}
