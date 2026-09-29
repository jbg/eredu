//! Complete architecture input survives the generic partition executor.
use super::*;
use eredu_architectures::partitioned_execution::{
    PipelinePartitionExecutor, PipelinePartitionUnitStrategy,
};
use eredu_runtime::*;
use std::borrow::Borrow;

struct TargetUnits(ParameterProviders<ResidentExpertProvider, Rows>);

impl PipelinePartitionUnitStrategy<TargetModel<NumericBackend>, NumericBackend, State>
    for TargetUnits
{
    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut TargetModel<NumericBackend>,
        address: ExecutionUnitAddress,
        unit: &mut Unit<NumericBackend>,
        hidden: &NumericTensor,
        state: &mut State,
        forward: &mut <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::ForwardContext,
        pass: ExpertPass,
        parallel: Option<&NumericParallelContext>,
        _communication: &PartitionCommunication<NumericBackend, G, R, I>,
        _communication_executor: &NumericContext,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error>
    where
        G: Borrow<u64>,
        R: Borrow<u64>,
        I: CommunicationTensorMetadata<NumericBackend>,
    {
        assert!(parallel.is_none());
        architecture.forward_unit_with_provider(
            address.group(),
            address.index(),
            unit,
            hidden,
            state,
            forward,
            pass,
            &mut self.0,
            context,
        )
    }
}

type Executor = PipelinePartitionExecutor<
    TargetModel<NumericBackend>,
    NumericBackend,
    State,
    ResidentUnitWindow<Unit<NumericBackend>>,
    NumericPartitionTensorAllocator,
    TargetUnits,
>;

fn units(
    architecture: &TargetModel<NumericBackend>,
    ctx: &NumericContext,
) -> Vec<Unit<NumericBackend>> {
    (0..specification().units.len())
        .map(|index| {
            let mut unit = architecture.construct_unit(index, ctx).unwrap();
            unit.visit_parameters_mut(&mut Parameters::default());
            unit
        })
        .collect()
}

fn execute(
    executor: &mut Executor,
    driver: &LayeredPartitionDriver,
    input: TargetInput<'_, NumericTensor>,
    state: &mut State,
    context: &NumericContext,
) -> Result<(NumericTensor, RequestContext<NumericTensor>), Error> {
    let communication = PartitionCommunication::new(
        CommunicationManifest::new(1, 0, Vec::new(), Vec::new()).unwrap(),
        Vec::<RealizedCommunicationGroup<u64>>::new(),
        Vec::<RealizedCommunicationRoute<u64>>::new(),
        NumericCommunicationMetadata(None),
    )
    .unwrap();
    let mut pass = <Executor as PartitionedGroupExecutor<
        TargetModel<NumericBackend>,
        NumericBackend,
        State,
        u64,
        u64,
        NumericCommunicationMetadata,
    >>::begin(executor, input, state, ExpertPass::Prefill, context)?;
    executor.execute_group(
        &mut pass,
        driver,
        state,
        &communication,
        context,
        context,
        &mut Observe::default(),
    )?;
    let (output, forward) = <Executor as PartitionedGroupExecutor<
        TargetModel<NumericBackend>,
        NumericBackend,
        State,
        u64,
        u64,
        NumericCommunicationMetadata,
    >>::finish(executor, pass, state, context)?;
    Ok((output, forward.request))
}

#[test]
fn qwen4_pipeline_executor_preserves_embedding_input_ids_and_visibility() {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let spec = specification();
    let architecture = model(spec.clone(), &context);
    let description = architecture.parameter_description(&context).unwrap();
    let boundary = <TargetModel<NumericBackend> as PartitionedLayeredArchitecture<
        NumericBackend,
        State,
    >>::boundary_schema(&architecture)
    .unwrap();
    let partition = ArchitecturePartition::from_description(
        &description,
        [(decoder::TARGET_EXECUTION_GROUP, 0..spec.units.len())],
        PartitionOwnership::new(true, true, ["embedding", "norm", "output"]).unwrap(),
        &spec.state_layout().unwrap(),
        &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..spec.units.len(),
        )]),
        (),
        boundary,
    )
    .unwrap();
    let driver = LayeredPartitionDriver::new(&partition, 0, 0..spec.units.len()).unwrap();
    let actual_units = units(&architecture, &context);
    let mut executor = Executor::new_with_unit_strategy(
        architecture,
        ResidentUnitWindow::new(actual_units),
        partition.units().collect(),
        None,
        NumericPartitionTensorAllocator,
        PipelineActivationDtype::Float32,
        TargetUnits(ParameterProviders {
            grouped: ResidentExpertProvider,
            rows: Rows::default(),
        }),
    )
    .unwrap();
    let reference = model(spec.clone(), &context);
    let reference_units = units(&reference, &context);
    let mut reference = LayerwiseRuntime::new(reference, ResidentUnitWindow::new(reference_units));
    let mut actual_state = state(&spec);
    let mut expected_state = state(&spec);
    let mut provider = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: Rows::default(),
    };
    let (ids, visible, embeddings) = input_values();
    for (start, end) in [(0, 18), (18, 19)] {
        let ids: Vec<_> = (0..2)
            .flat_map(|lane| ids[lane * 19 + start..lane * 19 + end].iter().copied())
            .collect();
        let visible: Vec<_> = (0..2)
            .flat_map(|lane| visible[lane * 19 + start..lane * 19 + end].iter().copied())
            .collect();
        let embeddings = embeddings.axis_slice(1, start, end);
        let angles: Vec<_> = (0..2 * (end - start) * 2)
            .map(|index| (index + start * 2) as f32 / 13.)
            .collect();
        let cosine = NumericTensor::new(
            [2, (end - start) as i32, 2],
            angles.iter().map(|a| a.cos()).collect(),
        );
        let sine = NumericTensor::new(
            [2, (end - start) as i32, 2],
            angles.iter().map(|a| a.sin()).collect(),
        );
        let input = || TargetInput {
            ids: Some(OriginalTokenIds::Host(&ids)),
            batch: 2,
            tokens: (end - start) as i32,
            embeddings: Some(&embeddings),
            visible: Some(TokenVisibility::Host(&visible)),
            rotary: Some(eredu_nn::RotaryPosition::Embeddings {
                cosine: &cosine,
                sine: &sine,
            }),
            position_delta: Some(3),
        };
        let (expected, _) = reference
            .forward_with_provider_and_observer_and_context(
                input(),
                &mut expected_state,
                ExpertPass::Prefill,
                &mut provider,
                &context,
                &mut Observe::default(),
            )
            .unwrap();
        let (actual, request) =
            execute(&mut executor, &driver, input(), &mut actual_state, &context).unwrap();
        assert_tensor_close(
            &actual,
            &expected,
            "generic partition preserves complete embedding input",
        );
        assert_eq!(request.ids(), ids);
        assert_eq!(request.visible(), visible);
        assert_eq!(request.position_delta(), 3);
        let eredu_nn::RotaryPosition::Embeddings {
            cosine: actual_cosine,
            sine: actual_sine,
        } = request.rotary()
        else {
            panic!("partition executor lost explicit rotary products");
        };
        assert_tensor_close(actual_cosine, &cosine, "retained cosine input");
        assert_tensor_close(actual_sine, &sine, "retained sine input");
        tensor_parallel::compare_state(&actual_state, &expected_state, &spec, 0, 1, &context);
    }

    let before = actual_state.clone();
    let error = execute(
        &mut executor,
        &driver,
        TargetInput {
            ids: None,
            batch: 2,
            tokens: 19,
            embeddings: Some(&embeddings),
            visible: Some(TokenVisibility::Host(&visible)),
            rotary: None,
            position_delta: None,
        },
        &mut actual_state,
        &context,
    )
    .unwrap_err();
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut missing_ids = false;
    while let Some(error) = cause {
        missing_ids |= matches!(
            error.downcast_ref::<RequestError>(),
            Some(RequestError::Tokens(
                eredu_architectures::qwen4_exp::ngram::NGramError::MissingTokenIds,
            ))
        );
        cause = error.source();
    }
    assert!(missing_ids, "missing IDs retain their typed cause: {error}");
    tensor_parallel::compare_state(&actual_state, &before, &spec, 0, 1, &context);
}
