use super::*;
use eredu_nn::{workspace::*, Parameter, ParameterSpec};
use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec, LayerwisePolicyForward};
use std::cell::Cell;

#[derive(Debug)]
struct Facts {
    missing_host: bool,
}
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 5,
            assumptions: "fixture output capacity and five bytes of simultaneous scratch".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok((!self.missing_host).then(|| WorkspaceHostBound {
            bytes: 3,
            assumptions: "three bytes of independent fixture staging".into(),
        }))
    }
}

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "WorkspaceTensor")]
struct Unit {
    weight: Parameter<WorkspaceTensor>,
    #[parameter(skip, retained_value)]
    helper: WorkspaceTensor,
}
impl Unit {
    fn build(size: i32, context: &WorkspaceContext) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::new(
                ParameterSpec::trainable("weight").unwrap(),
                WorkspaceTensor::unloaded_f32(&[2], context)?,
            ),
            helper: WorkspaceTensor::full_f32(0.75, &[size], context)?,
        })
    }
}

fn layout(units: usize) -> ExecutionUnitLayout {
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    ExecutionUnitLayout::new(&graph, [units]).unwrap()
}
struct Provider {
    layout: ExecutionUnitLayout,
    values: BTreeMap<ParameterId, WorkspaceTensor>,
    calls: Cell<usize>,
}
impl Provider {
    fn new(context: &WorkspaceContext, units: usize) -> Self {
        let storage = WorkspaceExistingStorage::new(Some(128), context);
        let value = WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap(),
            &storage,
            context,
        )
        .unwrap();
        Self {
            layout: layout(units),
            values: [(ParameterId::new("weight").unwrap(), value)].into(),
            calls: Cell::new(0),
        }
    }
}
impl WorkspaceLayerwiseParameters for Provider {
    fn layout(&self) -> &ExecutionUnitLayout {
        &self.layout
    }
    fn parameters(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        _: &WorkspaceContext,
    ) -> Result<BTreeMap<ParameterId, WorkspaceTensor>, Error> {
        assert_eq!(self.layout.address(ordinal), Some(address));
        self.calls.set(self.calls.get() + 1);
        Ok(self.values.clone())
    }
}

#[test]
fn every_prefill_and_decode_span_keeps_constructed_helpers_through_unit_completion() {
    let context = WorkspaceContext::new(Facts {
        missing_host: false,
    });
    let provider = Provider::new(&context, 2);
    let mut policy = WorkspaceLayerwisePolicy::new(&provider, layout(2)).unwrap();
    let initial = WorkspaceTensor::unloaded_f32(&[1, 1], &context).unwrap();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 2,
        max_output_tokens: 2,
        prefill_chunk_positions: 1,
        output: eredu_core::OutputDemand::LastPosition,
    };
    let mut builds = 0;
    let report = quote_inference_workspace_with_context(geometry, &context, |_| {
        context.begin_state_span(std::iter::empty::<&WorkspaceTensor>())?;
        let mut forward = LayerwisePolicyForward::<WorkspaceBackend, Unit, _>::begin(
            &mut policy,
            &initial,
            &context,
        )?;
        let mut last = initial.clone();
        for ordinal in 0..2 {
            let unit = forward
                .acquire(
                    ordinal,
                    provider.layout.address(ordinal).unwrap(),
                    |context| {
                        builds += 1;
                        Unit::build(ordinal as i32 + 2, context)
                    },
                )
                .map_err(|error| match error {
                    LayerwiseAcquireError::Architecture(error)
                    | LayerwiseAcquireError::Policy(error) => error,
                })?;
            // Existing padded weight storage remains separate from new helper
            // allocation; binding must preserve all 128 bytes of its backing.
            assert_eq!(
                context
                    .report(&[unit.weight.as_ref().clone()])?
                    .state
                    .unwrap()
                    .retained_bytes,
                Some(128)
            );
            last = unit.helper.square(&context)?;
            forward.complete(&last, std::iter::empty(), std::iter::empty())?;
        }
        forward.finish(&last)?;
        drop(forward);
        let report = context.report(&[])?;
        // Two scalar parameter seeds, two nonzero helpers and two forwards:
        // 48 tensor-output bytes, 30 scratch bytes and 18 disjoint host bytes.
        // Binding and unit Drop refund none of the constructor allocation.
        assert_eq!(report.total_bytes, Some(96));
        assert_eq!(report.operations.len(), 6);
        Ok::<_, Error>(report)
    })
    .unwrap();
    assert_eq!(builds, 8);
    assert_eq!(provider.calls.get(), 8);
    assert_eq!(report.transient().bytes(), Some(96));
    assert_eq!(report.first_gap(), None);
}

#[test]
fn layout_and_address_mismatch_reject_before_constructor_or_population() {
    let context = WorkspaceContext::new(Facts {
        missing_host: false,
    });
    let provider = Provider::new(&context, 2);
    assert!(WorkspaceLayerwisePolicy::new(&provider, layout(1)).is_err());
    let mut policy = WorkspaceLayerwisePolicy::new(&provider, layout(2)).unwrap();
    let built = Cell::new(false);
    let result = policy.acquire(
        0,
        provider.layout.address(1).unwrap(),
        |context| {
            built.set(true);
            Unit::build(2, context)
        },
        &context,
    );
    assert!(matches!(result, Err(LayerwiseAcquireError::Policy(_))));
    assert!(!built.get());
    assert_eq!(provider.calls.get(), 0);
}

#[test]
fn missing_parameter_rejects_before_forward_without_discarding_constructor_cost() {
    let context = WorkspaceContext::new(Facts {
        missing_host: false,
    });
    let mut provider = Provider::new(&context, 1);
    provider.values.clear();
    let mut policy = WorkspaceLayerwisePolicy::new(&provider, layout(1)).unwrap();
    context
        .begin_state_span(std::iter::empty::<&WorkspaceTensor>())
        .unwrap();
    let result = policy.acquire(
        0,
        provider.layout.address(0).unwrap(),
        |context| Unit::build(3, context),
        &context,
    );
    assert!(matches!(result, Err(LayerwiseAcquireError::Policy(_))));
    assert_eq!(provider.calls.get(), 1);
    let trace = context.report(&[]).unwrap();
    assert_eq!(trace.operations.len(), 2);
    assert_eq!(trace.total_bytes, Some(32));
}

#[test]
fn missing_constructor_host_fact_stays_unknown_after_metadata_completion() {
    let context = WorkspaceContext::new(Facts { missing_host: true });
    let provider = Provider::new(&context, 1);
    let mut policy = WorkspaceLayerwisePolicy::new(&provider, layout(1)).unwrap();
    context
        .begin_state_span(std::iter::empty::<&WorkspaceTensor>())
        .unwrap();
    let unit = policy
        .acquire(
            0,
            provider.layout.address(0).unwrap(),
            |context| Unit::build(2, context),
            &context,
        )
        .unwrap();
    let output = unit.helper.clone();
    policy
        .complete(
            0,
            provider.layout.address(0).unwrap(),
            unit,
            &output,
            std::iter::empty(),
            std::iter::empty(),
            &context,
        )
        .unwrap();
    <WorkspaceLayerwisePolicy<'_> as LayerwisePolicy<WorkspaceBackend, Unit>>::finish(
        &mut policy,
        &output,
        &context,
    )
    .unwrap();
    let trace = context.report(&[]).unwrap();
    assert_eq!(trace.tensor_buffers.total_bytes, Some(22));
    assert_eq!(trace.total_bytes, None);
    assert_eq!(trace.inference_transient_bytes(), None);
    assert_eq!(trace.unpriced_host_operations, [0, 1]);
}

#[test]
fn legacy_parameter_provider_does_not_allocate_through_strict_default() {
    let context = WorkspaceContext::new(Facts {
        missing_host: false,
    });
    let provider = Provider::new(&context, 1);
    assert!(matches!(
        provider.parameter_source(),
        Err(eredu_runtime::working_memory::WorkspaceParameterSourceError::CompanionUnavailable)
    ));
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(
        provider
            .parameters(0, provider.layout.address(0).unwrap(), &context)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(provider.calls.get(), 1);
}

impl WorkspaceFactMechanisms for Facts {
    type Error = std::convert::Infallible;
    fn operation_facts(&self, _: WorkspaceOperationView<'_>) -> Result<Option<WorkspaceOperationFacts>, Self::Error> { panic!("no constructor may run before source qualification") }
    fn write_operation_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceEffectDestination<'_>) -> Result<Option<WorkspaceOperationFacts>, Self::Error> { unreachable!() }
    fn host_facts(&self, _: WorkspaceOperationView<'_>) -> Result<Option<WorkspaceHostFacts>, Self::Error> { unreachable!() }
    fn write_host_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceHostDestination<'_>) -> Result<Option<WorkspaceHostFacts>, Self::Error> { unreachable!() }
}

#[test]
fn checked_acquire_refuses_legacy_population_before_constructor() {
    let context=WorkspaceContext::new_recording_facts(Facts {missing_host:false});
    let provider=Provider::new(&context,1);
    let mut policy=WorkspaceLayerwisePolicy::new(&provider,layout(1)).unwrap();
    let built=Cell::new(false);
    let result=policy.acquire(0,provider.layout.address(0).unwrap(),|context| {
        built.set(true);
        Unit::build(2,context)
    },&context);
    assert!(matches!(result,Err(LayerwiseAcquireError::Policy(_))));
    assert!(!built.get());
    assert_eq!(provider.calls.get(),0);
}
