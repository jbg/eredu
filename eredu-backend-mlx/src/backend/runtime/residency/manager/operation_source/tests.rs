use super::*;
use eredu_checkpoint::{
    recipe::RecipeCatalog,
    store::{StoreError, TensorMetadata, TensorSelection},
    StoredDtype,
};
use eredu_core::residency::{OffloadConfig, OffloadUnitSpec, ResidencyPolicy};
use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec};

struct Catalog;
impl RecipeCatalog for Catalog {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        Ok(TensorMetadata {
            name: key.into(),
            logical_shape: vec![1],
            physical_shape: vec![1],
            stored_dtype: StoredDtype::F32,
            encoded_byte_len: 4,
            backing_shard: None,
        })
    }
}
fn source() -> (ResidencyController, Vec<OffloadUnitId>, ExecutionUnitLayout) {
    let ids = ["A", "B", "tail"]
        .map(|id| OffloadUnitId::new(id).unwrap())
        .to_vec();
    let units = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let owner =
                WeightBinding::new("owner", format!("physical.{i}"), TensorSelection::Full, 4)
                    .unwrap()
                    .with_logical_target(format!("logical.{i}"))
                    .unwrap();
            let mut rows = vec![owner];
            if i < 2 {
                rows.push(WeightBinding::alias("other", format!("logical.{}", 1 - i), 4).unwrap());
            }
            OffloadUnit::new(id.clone(), rows).unwrap()
        })
        .collect::<Vec<_>>();
    let plan = OffloadPlan::new(
        OffloadConfig::default(),
        ids.iter().map(|id| {
            OffloadUnitSpec::new(id.clone(), 4, ResidencyPolicy::Windowed, MemoryTier::Disk)
                .unwrap()
        }),
    )
    .unwrap();
    let controller = ResidencyController::new(&Catalog, plan, units).unwrap();
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root("first"),
            ExecutionGroupSpec::with_dependencies("last", ["first"]),
        ],
        "last",
    )
    .unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [2, 1]).unwrap();
    (controller, ids, layout)
}

#[test]
fn actual_cyclic_sources_and_group_cut_windows_produce_scalar_rows_without_source_clones() {
    let (controller, ids, layout) = source();
    let before = controller
        .units()
        .map(|unit| unit as *const OffloadUnit)
        .collect::<Vec<_>>();
    let source = OriginalResidencySource::prepare(
        &controller,
        &ids,
        &layout,
        NonZeroUsize::new(99).unwrap(),
    )
    .unwrap();
    assert_eq!(source.controller_units, 3);
    assert_eq!(
        source
            .windows()
            .iter()
            .map(|row| (row.request_start, row.request_end))
            .collect::<Vec<_>>(),
        [(0, 2), (1, 2), (2, 3)]
    );
    let windows = source.window_owner();
    assert!(Arc::ptr_eq(&windows.values, &source.window_owner().values));
    assert!(std::ptr::eq(windows.as_slice(), source.windows()));
    let weak = Arc::downgrade(&windows.values);
    drop(windows);
    assert!(weak.upgrade().is_some());
    assert_eq!(
        source
            .windows()
            .iter()
            .map(|row| row.requested)
            .collect::<Vec<_>>(),
        [2, 1, 1]
    );
    assert_eq!(
        source
            .windows()
            .iter()
            .map(|row| row.units)
            .collect::<Vec<_>>(),
        [2, 2, 1]
    );
    for row in &source.windows()[..2] {
        assert_eq!(
            (
                row.bindings,
                row.physical_bindings,
                row.aliases,
                row.local_aliases
            ),
            (4, 2, 2, 0)
        );
        assert_eq!(
            (
                row.physical_bytes,
                row.recipe_pending,
                row.recipe_materializations
            ),
            (8, 2, 0)
        );
        assert_eq!(
            (
                row.unit_id_bytes,
                row.binding_name_bytes,
                row.alias_owner_name_bytes
            ),
            (2, 20, 10)
        );
    }
    let cycle = source.windows()[0];
    assert_eq!(
        cycle.declaration_clone_bytes,
        4 * size_of::<WeightBinding>() + 76
    );
    assert_eq!(cycle.declaration_clone_allocations, 12); // Two binding Vecs, ten nonempty Strings.
    assert_eq!(source.windows()[2].physical_bytes, 4);
    assert_eq!(
        controller
            .units()
            .map(|unit| unit as *const OffloadUnit)
            .collect::<Vec<_>>(),
        before
    );
    assert!(source.retained_control_bytes().unwrap() > 0);
    let constructor =
        OriginalResidencySource::constructor_storage_bytes(controller.units().len(), layout.len())
            .unwrap();
    assert!(
        constructor
            >= source.retained_control_bytes().unwrap()
                + (controller.units().len() * size_of::<ResidencyClosureSlot>()) as u64
    );
    assert!(OriginalResidencySource::constructor_storage_bytes(usize::MAX, layout.len()).is_none());
    assert!(OriginalResidencySource::constructor_storage_bytes(
        controller.units().len(),
        usize::MAX
    )
    .is_none());
    let retained = source.window_owner();
    drop(source);
    assert!(weak.upgrade().is_some());
    assert_eq!(retained[1].request_start, 1);
    drop(retained);
    assert!(weak.upgrade().is_none());
}

#[test]
fn repeated_roots_have_distinct_request_payload_but_one_canonical_owner_population() {
    let (controller, ids, _) = source();
    let mut scratch = vec![ResidencyClosureSlot::default(); 3];
    let row = WindowPopulation::collect(
        &controller,
        &[
            ids[0].clone(),
            ids[0].clone(),
            ids[0].clone(),
            ids[0].clone(),
        ],
        &mut scratch,
    )
    .unwrap();
    assert_eq!(
        (
            row.requested,
            row.requested_id_bytes,
            row.units,
            row.physical_bindings
        ),
        (4, 4, 2, 2)
    );
    // Cold closure does not change the existing runtime duplicate-batch refusal.
    assert!(WindowPopulation::collect(
        &controller,
        &[OffloadUnitId::new("foreign").unwrap()],
        &mut scratch
    )
    .is_err());
}

#[test]
fn actual_recipe_selection_and_companion_clone_layout_includes_every_owned_field() {
    use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeDtype};
    let recipe = DerivedWeightRecipe::View {
        input: Box::new(DerivedWeightRecipe::source(
            "key",
            TensorSelection::Indices {
                axis: 0,
                indices: vec![2, 1, 0],
            },
        )),
        dtype: RecipeDtype::Other("packed".into()),
        shape: vec![3],
    };
    // This declaration's custom dtype will require its ordinary validator.
    // Source-clone storage is still real on validation/failure paths.
    let binding = WeightBinding::from_recipe("weight", recipe, 12)
        .unwrap()
        .with_quantization_companions("s", Some("b".into()))
        .unwrap();
    let shape = DeclarationCloneShape::bindings(std::slice::from_ref(&binding)).unwrap();
    assert_eq!(
        shape.payload_bytes,
        size_of::<WeightBinding>() + size_of::<DerivedWeightRecipe>() + 20 + 4 * size_of::<usize>()
    );
    assert_eq!(shape.allocations, 10);
    assert_eq!(binding.checkpoint_key(), "key");
    assert_eq!(binding.quantization_companions().unwrap().scale(), "s");
}
