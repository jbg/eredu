//! Actual resident construction versus the original fallible collector.
use super::*;
use std::sync::Mutex;

type State = DeviceState<FakeBackend, FakeLayerState>;
type Events = Arc<Mutex<Vec<Event>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    Graph,
    Count(usize),
    Build(usize, usize),
    Panic(usize, usize),
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Graph,
    Count(usize),
    Build(usize, usize),
    Failed(Fault),
    UnitDrop(usize, usize),
    ArchitectureDrop,
    ErrorDrop(Fault),
}
#[derive(Debug)]
struct Failure {
    fault: Fault,
    events: Events,
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "construction failure: {:?}", self.fault)
    }
}
impl std::error::Error for Failure {}
impl Drop for Failure {
    fn drop(&mut self) {
        self.events
            .lock()
            .unwrap()
            .push(Event::ErrorDrop(self.fault));
    }
}

struct Unit<const N: usize> {
    group: usize,
    index: usize,
    value: u64,
    payload: [u8; N],
    events: Events,
}
impl<const N: usize> Drop for Unit<N> {
    fn drop(&mut self) {
        self.events
            .lock()
            .unwrap()
            .push(Event::UnitDrop(self.group, self.index));
    }
}
impl<const N: usize> Parameterized<FakeTensor> for Unit<N> {
    fn visit_parameter_sources<'a, V: eredu_nn::ParameterSourceVisitor<'a, FakeTensor>>(&'a self, _: &mut V) -> Result<(), eredu_nn::ParameterSourceError> {
 let mut __source_result = Ok(());

 __source_result
}
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, FakeTensor>>(&'a mut self, _: &mut V) {}
    fn set_trainable(&mut self, _: bool) {}
}
struct Architecture<const N: usize> {
    counts: Vec<usize>,
    graph: ExecutionGraph,
    fault: Option<Fault>,
    events: Events,
    static_modules: FakeOperator,
}
impl<const N: usize> Architecture<N> {
    fn new(counts: &[usize], fault: Option<Fault>, events: &Events) -> Self {
        Self {
            counts: counts.to_vec(),
            graph: ExecutionGraph::chain((0..counts.len()).map(|group| format!("g{group}"))).unwrap(),
            fault,
            events: events.clone(),
            static_modules: FakeOperator,
        }
    }
    fn check(&self, fault: Fault) -> Result<(), Failure> {
        if self.fault == Some(fault) {
            self.events.lock().unwrap().push(Event::Failed(fault));
            Err(Failure {
                fault,
                events: self.events.clone(),
            })
        } else {
            Ok(())
        }
    }
}
impl<const N: usize> Drop for Architecture<N> {
    fn drop(&mut self) {
        self.events.lock().unwrap().push(Event::ArchitectureDrop);
    }
}
impl<const N: usize> ArchitectureParameters<FakeBackend> for Architecture<N> {
    type DefinitionError = Failure;
    fn state_layout(&self, _metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<StateLayout, Failure> {
        unreachable!("construction does not realize state")
    }
    fn state_identity(
        &self,
        _: &eredu_runtime::PartitionState,
        _: eredu_core::cache::PromptCacheTopology,
        _metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Failure> {
        unreachable!("construction does not derive state identity")
    }
    fn parameter_description(&self, _: &()) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Failure> {
        unreachable!("construction does not traverse parameter descriptions")
    }
    fn visit_static_parameters<V: StaticParameterVisitor<FakeBackend>>(
        &self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        unreachable!("construction does not visit static parameters")
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<FakeBackend>>(
        &mut self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        unreachable!("construction does not visit static parameters")
    }
}
impl<const N: usize> LayeredArchitecture<FakeBackend, State> for Architecture<N> {
    type Input<'a> = &'a FakeTensor;
    type StaticModules = FakeOperator;
    type Unit = Unit<N>;
    type ForwardContext = ();
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Failure;

    fn inference_input_shape(_: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Failure> {
        unreachable!("construction does not inspect a forward input")
    }
    fn group_transport(&self, _: usize) -> ArchitectureGroupTransport {
        unreachable!("construction does not select transport")
    }
    fn primary_execution_group(&self) -> &str {
        "g0"
    }
    fn state_partition_plan(
        &self,
        _: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        unreachable!("construction does not partition state")
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Failure> {
        self.events.lock().unwrap().push(Event::Graph);
        self.check(Fault::Graph)?;
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(&self.graph))
    }
    fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Failure> {
        self.events.lock().unwrap().push(Event::Count(group));
        self.check(Fault::Count(group))?;
        Ok(self.counts[group])
    }
    fn unit_path(&self, _: usize, _: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Failure> {
        unreachable!("construction leaves unit paths to the architecture")
    }
    fn static_modules(&self) -> &FakeOperator {
        &self.static_modules
    }
    fn static_modules_mut(&mut self) -> &mut FakeOperator {
        &mut self.static_modules
    }
    fn build_unit(&self, group: usize, index: usize, _: &()) -> Result<Unit<N>, Failure> {
        self.events.lock().unwrap().push(Event::Build(group, index));
        if self.fault == Some(Fault::Panic(group, index)) {
            self.events
                .lock()
                .unwrap()
                .push(Event::Failed(Fault::Panic(group, index)));
            panic!("unit construction panic");
        }
        self.check(Fault::Build(group, index))?;
        let value = (group as u64 + 1) * 17 + index as u64 + 1;
        Ok(Unit {
            group,
            index,
            value,
            payload: [value as u8; N],
            events: self.events.clone(),
        })
    }
    fn begin_forward<'a>(
        &mut self,
        _: Self::Input<'a>,
        _: &mut State,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, ()>, Failure> {
        unreachable!("construction test does not execute a forward")
    }
    fn begin_execution_group(
        &mut self,
        _: usize,
        _: &FakeTensor,
        _: &[&FakeTensor],
        _: &mut State,
        _: &mut (),
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        unreachable!("construction test does not execute a forward")
    }
    fn forward_unit(
        &mut self,
        _: usize,
        _: usize,
        _: &mut Unit<N>,
        _: &FakeTensor,
        _: &mut State,
        _: &mut (),
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        unreachable!("construction test does not execute a forward")
    }
    fn select_readout_positions(
        &self,
        _: &FakeTensor,
        _: &(),
        _: eredu_core::OutputDemand,
        _: &(),
    ) -> Result<Option<FakeTensor>, Failure> {
        unreachable!("construction test does not execute a forward")
    }
    fn finish_forward(
        &mut self,
        _: &FakeTensor,
        _: &mut State,
        _: &(),
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        unreachable!("construction test does not execute a forward")
    }
    fn retained_context_values<'a>(
        &'a self,
        _: &'a (),
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}

// Success field order matches ResidentRuntime. Only the unrelated observation
// binding/phantom is omitted; neither runs an architecture/unit destructor.
struct Reference<const N: usize> {
    _architecture: Architecture<N>,
    _graph: ExecutionGraph,
    units: Vec<Vec<Unit<N>>>,
}
fn reference<const N: usize>(architecture: Architecture<N>) -> Result<Reference<N>, Failure> {
    let context = &();
    // The old constructor's graph/count/build/collect body is kept independently.
    let graph = architecture.execution_graph()?.into_owned();
    let mut units = Vec::with_capacity(graph.groups().len());
    for group in 0..graph.groups().len() {
        let count = architecture.group_unit_count(group, None)?;
        units.push(
            (0..count)
                .map(|index| architecture.build_unit(group, index, context))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(Reference {
        _architecture: architecture,
        _graph: graph,
        units,
    })
}
fn construct<const N: usize>(
    architecture: Architecture<N>,
) -> Result<ResidentRuntime<Architecture<N>, FakeBackend, State>, Failure> {
    ResidentRuntime::new(architecture, &())
}
fn events() -> Events {
    Arc::new(Mutex::new(Vec::new()))
}
fn snapshot(events: &Events) -> Vec<Event> {
    events.lock().unwrap().clone()
}
fn values<const N: usize>(units: &[Vec<Unit<N>>]) -> Vec<Vec<u64>> {
    units
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|unit| {
                    assert!(unit.payload.iter().all(|value| *value == unit.value as u8));
                    unit.value
                })
                .collect()
        })
        .collect()
}

#[test]
fn resident_construction_preserves_values_order_and_empty_groups() {
    let current = events();
    let old = events();
    let runtime = construct(Architecture::<2048>::new(&[2, 0, 3], None, &current)).unwrap();
    let reference = reference(Architecture::<2048>::new(&[2, 0, 3], None, &old)).unwrap();
    assert_eq!(
        values(runtime.units()),
        [vec![18, 19], vec![], vec![52, 53, 54]]
    );
    assert_eq!(values(runtime.units()), values(&reference.units));
    assert_eq!(snapshot(&current), snapshot(&old));
    assert_eq!(
        snapshot(&current),
        [
            Event::Graph,
            Event::Count(0),
            Event::Build(0, 0),
            Event::Build(0, 1),
            Event::Count(1),
            Event::Count(2),
            Event::Build(2, 0),
            Event::Build(2, 1),
            Event::Build(2, 2)
        ]
    );
    drop(runtime);
    drop(reference);
    assert_eq!(snapshot(&current), snapshot(&old));
    assert!(snapshot(&current).ends_with(&[
        Event::ArchitectureDrop,
        Event::UnitDrop(0, 0),
        Event::UnitDrop(0, 1),
        Event::UnitDrop(2, 0),
        Event::UnitDrop(2, 1),
        Event::UnitDrop(2, 2),
    ]));
}

#[test]
fn resident_construction_preserves_error_and_prefix_retirement() {
    for fault in [
        Fault::Graph,
        Fault::Count(0),
        Fault::Count(1),
        Fault::Count(2),
        Fault::Build(0, 0),
        Fault::Build(0, 1),
        Fault::Build(2, 0),
        Fault::Build(2, 1),
    ] {
        let current = events();
        let old = events();
        let error = construct(Architecture::<2048>::new(&[2, 0, 3], Some(fault), &current))
            .err()
            .expect("injected constructor failure");
        let before = reference(Architecture::<2048>::new(&[2, 0, 3], Some(fault), &old))
            .err()
            .expect("old constructor failure");
        assert_eq!(error.fault, fault);
        assert_eq!(error.to_string(), before.to_string());
        assert_eq!(snapshot(&current), snapshot(&old), "{fault:?}");
        assert_eq!(snapshot(&current).last(), Some(&Event::ArchitectureDrop));
        assert!(!snapshot(&current)
            .iter()
            .any(|e| matches!(e, Event::ErrorDrop(_))));
        if fault == Fault::Build(2, 1) {
            assert!(snapshot(&current).ends_with(&[
                Event::UnitDrop(2, 0),
                Event::UnitDrop(0, 0),
                Event::UnitDrop(0, 1),
                Event::ArchitectureDrop,
            ]));
        }
        drop(error);
        drop(before);
        assert_eq!(snapshot(&current), snapshot(&old));
        assert_eq!(snapshot(&current).last(), Some(&Event::ErrorDrop(fault)));
    }
}

fn compare_growth<const N: usize>() {
    for count in 0..=65 {
        let current = events();
        let old = events();
        let runtime = construct(Architecture::<N>::new(&[count], None, &current)).unwrap();
        let reference = reference(Architecture::<N>::new(&[count], None, &old)).unwrap();
        assert_eq!(
            runtime.units()[0].capacity(),
            reference.units[0].capacity(),
            "N={N}, count={count}"
        );
        assert_eq!(values(runtime.units()), values(&reference.units));
        if count == 0 {
            assert_eq!(runtime.units()[0].capacity(), 0);
        }
        drop(runtime);
        drop(reference);
        assert_eq!(snapshot(&current), snapshot(&old));
    }
}
#[test]
fn resident_construction_preserves_small_and_large_unit_capacity_growth() {
    assert!(std::mem::size_of::<Unit<0>>() <= 1024);
    assert!(std::mem::size_of::<Unit<2048>>() > 1024);
    compare_growth::<0>();
    compare_growth::<2048>();
}

#[test]
fn resident_construction_preserves_unwind_prefix_retirement() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    for fault in [Fault::Panic(0, 0), Fault::Panic(0, 1), Fault::Panic(2, 1)] {
        let current = events();
        let old = events();
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = construct(Architecture::<2048>::new(&[2, 0, 3], Some(fault), &current));
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = reference(Architecture::<2048>::new(&[2, 0, 3], Some(fault), &old));
        }))
        .is_err());
        assert_eq!(snapshot(&current), snapshot(&old), "{fault:?}");
        assert_eq!(snapshot(&current).last(), Some(&Event::ArchitectureDrop));
    }
}
