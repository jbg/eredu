use super::*;
use crate::preparation_selection::{
    select_preparation,
    tests::{inspected_config, prediction_config, BoundedIndependentAdapter},
};
use eredu_nn::{workspace::*, ParameterMetadata, ParameterMetadataView, ParameterVisitor};
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, WorkingMemoryPool},
    NormalizedLoadRequest,
};
use std::{cell::Cell, convert::Infallible, num::NonZeroU32};

#[derive(Debug)]
struct MissingFacts;
impl WorkspaceMechanisms for MissingFacts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("counted source entered ordinary facts")
    }
}
impl WorkspaceFactMechanisms for MissingFacts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("missing facts cannot emit")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("missing host facts cannot emit")
    }
}

pub(crate) struct ProjectedFixtureParameters;
impl WorkspacePredictionParameterSource for ProjectedFixtureParameters {
    type Context<'a> = &'a Cell<usize>;
    fn bind<U: Parameterized<WorkspaceTensor>>(
        calls: &mut Self::Context<'_>,
        ordinal: usize,
        _source: &U,
        local: &mut U,
        tasks: &[ReplicatedTextMaterializationTask],
        layout: Option<&LocalModelLayout>,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        assert_eq!(ordinal, calls.get());
        assert!(!tasks.is_empty());
        assert!(layout.is_some());
        struct Values<'a> {
            context: &'a WorkspaceContext,
            values: Vec<(String, WorkspaceTensor)>,
            error: Option<Error>,
        }
        impl<'v> ParameterVisitor<'v, WorkspaceTensor> for Values<'_> {



            fn visit(
                &mut self,
                metadata: ParameterMetadataView<'_>,
                value: &'v WorkspaceTensor,
            ) {
                if self.error.is_some() {
                    return;
                }
                let result = (|| {
                    self.context.reserve_metadata_vec(&mut self.values, 1)?;
                    let name = self
                        .context
                        .metadata_string(format_args!("{}", metadata.id().as_str()))?;
                    // Fixture-owned real initialized metadata backing, preserving
                    // the selected packed/companion dtype and source shape.
                    let weight = WorkspaceTensor::initialized(
                        value.shape(),
                        value.layout().dtype(),
                        self.context,
                    )?;
                    self.values.push((name, weight));
                    Ok::<_, Error>(())
                })();
                self.error = result.err();
            }
        }
        let mut values = Values {
            context,
            values: Vec::new(),
            error: None,
        };
        local.visit_parameters(&mut values);
        if let Some(cause) = values.error {
            return Err(cause);
        }
        assert!(!values.values.is_empty());
        let mut rows = context.metadata_vec(values.values.len())?;
        for (name, value) in &values.values {
            rows.push(eredu_runtime::parameter::PreparedParameterBinding::new(
                name,
                value.clone(),
            ));
        }
        context.charge_metadata(
            eredu_runtime::parameter::prepared_parameter_binding_control_bytes::<
                WorkspaceTensor,
                WorkspaceTensor,
                Error,
            >(0)
            .unwrap(),
        )?;
        eredu_runtime::parameter::bind_prepared_parameter_values(
            local,
            &mut rows,
            |_| false,
            |old, new| {
                if old.shape() != new.shape()
                    || old.layout().dtype() != new.layout().dtype()
                    || !old.same_context(new)
                {
                    return Err(
                        context.metadata_error(format_args!("fixture source changed layout"))
                    );
                }
                Ok(())
            },
            |old, new| *old = new,
        )
        .map_err(|cause| context.metadata_source(cause))?;
        calls.set(calls.get() + 1);
        Ok(())
    }
}

#[test]
fn selected_prediction_materializer_preserves_current_backing_and_refuses_foreign_trace() {
    let (_directory, inspection) = inspected_config(prediction_config());
    let request = NormalizedLoadRequest::default();
    let selected =
        select_preparation(&inspection, &request, &BoundedIndependentAdapter::default()).unwrap();
    let capacity = 1 << 27;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let funding = pool
        .prepare_workspace_metadata(&execution, capacity)
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(MissingFacts, funding.clone()).unwrap();
    let foreign = WorkspaceContext::new_with_metadata_funding(MissingFacts, funding).unwrap();
    let prepared = || {
        crate::prediction_extension::prepare_replicated_prediction_extension::<WorkspaceBackend>(
            selected.prediction_extension().unwrap(),
            selected
                .text_realization()
                .auxiliary_materialization_tasks(),
            &context,
            &context,
        )
        .unwrap()
    };
    let policy = LayerCachePolicy::CompressedLatentRotary {
        attention: eredu_core::AttentionPolicy::Full,
        latent_dim: NonZeroU32::new(4).unwrap(),
        rotary_dim: NonZeroU32::new(2).unwrap(),
    };
    let mut cache = WorkspaceCompressedCache::new(
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(4).unwrap(),
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(4).unwrap(),
        &context,
    )
    .unwrap();
    cache
        .append(
            CompressedAttentionState {
                latent: WorkspaceTensor::initialized(&[1, 3, 4], WorkspaceDtype::Float32, &context)
                    .unwrap(),
                rotary: WorkspaceTensor::initialized(&[1, 3, 2], WorkspaceDtype::Float32, &context)
                    .unwrap(),
            },
            &context,
        )
        .unwrap();
    cache.validate_projected_policy(&policy, &context).unwrap();
    let current = WorkspacePredictionState::Sequential(vec![cache]);
    let calls = Cell::new(0);
    let error = match prepared()
        .materialize_workspace::<ProjectedFixtureParameters>(&calls, &current, &foreign)
    {
        Ok(_) => panic!("foreign trace must refuse before binding"),
        Err(error) => error,
    };
    assert_eq!(calls.get(), 0);
    let materialized = prepared()
        .materialize_workspace::<ProjectedFixtureParameters>(&calls, &current, &context)
        .unwrap();
    assert_eq!(calls.get(), 1);
    type Model = crate::deepseek::v3::Model<WorkspaceBackend>;
    let executor = Model::pair_prediction_extension(materialized).unwrap();
    let mut lane = current
        .prepare_current_lane::<Model, _, ProjectedFixtureParameters>(&executor, &context)
        .unwrap();
    assert_eq!(executor.prefill_frontier(&mut lane).unwrap(), 3);
    let WorkspacePredictionState::Sequential(source) = &current else {
        unreachable!()
    };
    assert_eq!(source[0].capacity(), 4);
    let prior = source[0].retained_values().collect::<Vec<_>>();
    let copied = lane[0].retained_values().collect::<Vec<_>>();
    assert_eq!(prior.len(), copied.len());
    for (a, b) in prior.iter().zip(&copied) {
        let single = context.report_scalars(std::slice::from_ref(*a)).unwrap().closing_storage;
        let paired = context.report_scalars(&[(*a).clone(), (*b).clone()]).unwrap().closing_storage;
        assert_eq!(single, paired);
    }
    drop(copied);
    drop(prior);
    let before = pool.used_bytes().unwrap();
    lane[0]
        .append(
            CompressedAttentionState {
                latent: WorkspaceTensor::initialized(&[1, 2, 4], WorkspaceDtype::Float32, &context)
                    .unwrap(),
                rotary: WorkspaceTensor::initialized(&[1, 2, 2], WorkspaceDtype::Float32, &context)
                    .unwrap(),
            },
            &context,
        )
        .unwrap();
    assert_eq!(lane[0].offset(), 5);
    assert_eq!(source[0].offset(), 3);
    assert!(pool.used_bytes().unwrap() > before);
    drop(lane);
    drop(executor);
    drop(current);
    drop(context);
    drop(foreign);
    assert!(
        pool.used_bytes().unwrap() > 0,
        "escaped source failure lost its metadata custody"
    );
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
