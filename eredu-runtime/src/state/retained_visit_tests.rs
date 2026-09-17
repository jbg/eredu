use super::*;
use eredu_nn::workspace::*;
use std::{cell::Cell, convert::Infallible};
#[derive(Debug)]
struct NoWork;
impl WorkspaceMechanisms for NoWork {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        panic!("source visiting cannot create numerical work")
    }
}
fn value(context: &WorkspaceContext) -> WorkspaceTensor {
    WorkspaceTensor::existing(
        WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap(),
        context,
    )
    .unwrap()
}
struct LegacyLayer {
    value: WorkspaceTensor,
    iterators: Cell<usize>,
}
impl RuntimeLayerState<WorkspaceBackend> for LegacyLayer {
    type RetainedValues<'a> = std::iter::Once<&'a WorkspaceTensor>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.iterators.set(self.iterators.get() + 1);
        std::iter::once(&self.value)
    }
}
#[test]
fn borrowed_layer_callback_preserves_the_legacy_iterator_default() {
    let context = WorkspaceContext::new(NoWork);
    let layer = LegacyLayer {
        value: value(&context),
        iterators: Cell::new(0),
    };
    let mut count = 0;
    layer.visit_retained_values(&mut |tensor| {
        assert!(std::ptr::eq(tensor, &layer.value));
        count += 1;
    });
    assert_eq!(count, 1);
    assert_eq!(layer.iterators.get(), 1);
}
struct DirectLayer {
    values: [WorkspaceTensor; 2],
    visits: Cell<usize>,
}
impl RuntimeLayerState<WorkspaceBackend> for DirectLayer {
    type RetainedValues<'a> = std::iter::Empty<&'a WorkspaceTensor>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        panic!("whole-state access must use the provided direct override")
    }
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&WorkspaceTensor)) {
        self.visits.set(self.visits.get() + 1);
        for value in &self.values {
            visitor(value);
        }
    }
}
#[test]
fn device_whole_state_uses_each_direct_layer_override_once_without_iterator_construction() {
    let context = WorkspaceContext::new(NoWork);
    let layout = StateLayout::new(
        LayerSchedule::new(
            2,
            vec![LayerCachePolicy::key_value(eredu_core::AttentionPolicy::Full, 1, 2).unwrap(); 2],
        )
        .unwrap(),
    )
    .unwrap();
    let state = DeviceState::<WorkspaceBackend, DirectLayer>::create(layout, |_, _| {
        Ok::<_, Infallible>(DirectLayer {
            values: [value(&context), value(&context)],
            visits: Cell::new(0),
        })
    })
    .unwrap();
    let expected = [
        &state.as_ref()[0].values[0],
        &state.as_ref()[0].values[1],
        &state.as_ref()[1].values[0],
        &state.as_ref()[1].values[1],
    ];
    let mut count = 0;
    state
        .visit_all_retained_values(&mut |tensor| {
            assert!(std::ptr::eq(tensor, expected[count]));
            count += 1;
        })
        .unwrap();
    assert_eq!(count, 4);
    assert!(state.as_ref().iter().all(|layer| layer.visits.get() == 1));
    let empty = DeviceState::<WorkspaceBackend, DirectLayer>::stateless();
    empty
        .visit_all_retained_values(&mut |_| panic!("stateless owner has no values"))
        .unwrap();
}
