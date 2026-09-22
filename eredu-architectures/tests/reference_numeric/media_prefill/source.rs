//! Genuine host-source construction for the numerical media fixture.
use super::*;
use eredu_nn::workspace::*;
use eredu_runtime::input::host::{
    HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};

enum Values {
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
}
struct Slot {
    shape: Vec<usize>,
    values: Values,
}
impl Slot {
    fn new(value: &NumericTensor) -> Self {
        use eredu_core::checkpoint::TensorDtype;
        let values = match value.dtype {
            TensorDtype::U32 => Values::U32(value.data.iter().map(|v| *v as u32).collect()),
            TensorDtype::I32 => Values::I32(value.data.iter().map(|v| *v as i32).collect()),
            TensorDtype::Bool => Values::Bool(value.data.iter().map(|v| *v != 0.0).collect()),
            TensorDtype::F32 => Values::F32(value.data.clone()),
            _ => panic!("numerical host fixture scalar dtype"),
        };
        Self {
            shape: value
                .shape
                .iter()
                .map(|n| usize::try_from(*n).unwrap())
                .collect(),
            values,
        }
    }
    fn metadata(key: eredu_core::InputMetadataKey, value: &NumericTensor) -> Self {
        // The numerical backend represents integer and Boolean host reads in
        // f32 storage; construct the canonical host scalar type explicitly.
        let mut slot = Self::new(value);
        match key {
            eredu_core::InputMetadataKey::PatchGrid
            | eredu_core::InputMetadataKey::PatchPositions => {
                assert!(value
                    .data
                    .iter()
                    .all(|v| v.is_finite() && *v == (*v as i32) as f32));
                slot.values = Values::I32(value.data.iter().map(|v| *v as i32).collect());
            }
            eredu_core::InputMetadataKey::AudioMask => {
                assert!(value.data.iter().all(|v| *v == 0.0 || *v == 1.0));
                slot.values = Values::Bool(value.data.iter().map(|v| *v != 0.0).collect());
            }
            _ => {}
        }
        slot
    }
    fn view(&self) -> HostTensorView<'_> {
        HostTensorView {
            shape: &self.shape,
            values: match &self.values {
                Values::U32(v) => HostTensorValues::U32(v),
                Values::I32(v) => HostTensorValues::I32(v),
                Values::F32(v) => HostTensorValues::F32(v),
                Values::Bool(v) => HostTensorValues::Bool(v),
            },
        }
    }
}

#[derive(Debug)]
struct MetadataFacts;
pub(super) fn unfunded_metadata() -> WorkspaceContext {
    WorkspaceContext::new(MetadataFacts)
}
impl WorkspaceMechanisms for MetadataFacts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
impl WorkspaceFactMechanisms for MetadataFacts {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
}

pub(super) fn prepare<A, D>(
    session: &Session<A, D>,
    admission: &A::AdmissionConfig,
    input: &Input,
    sources: &eredu_architectures::prepared_sources::PreparedModelSources,
    geometry: eredu_core::InferenceGeometry,
) -> Result<
    eredu_runtime::media_prefill::PreparedMediaPrefill<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        State,
    >,
    Error,
>
where
    A: CompositeMediaIngressArchitecture<NumericBackend, State, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    D: MediaTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    // These caller-owned values precede source construction. The canonical
    // compiler reserves its independent buffers before copying them.
    let payloads = input
        .parts()
        .iter()
        .map(|part| Slot::new(part.payload().value()))
        .collect::<Vec<_>>();
    let metadata = input
        .parts()
        .iter()
        .map(|part| {
            part.metadata()
                .iter()
                .map(|(key, value)| (*key, Slot::metadata(*key, value)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let views = metadata
        .iter()
        .map(|slots| {
            slots
                .iter()
                .map(|(key, slot)| (*key, slot.view()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let parts = input
        .parts()
        .iter()
        .enumerate()
        .map(|(i, part)| HostInputPart {
            modality: part.modality(),
            kind: part.payload().kind(),
            payload: payloads[i].view(),
            metadata: &views[i],
            extents: part.extents(),
        })
        .collect::<Vec<_>>();
    let pool = crate::memory_fixture::ledger(1 << 30, 0).map_err(Error::backend)?;
    let host = pool
        .compile_prepared_host_input(
            PreparedHostInputPlan::prepare(&parts).map_err(Error::backend)?,
        )
        .map_err(Error::backend)?;
    let semantics = sources
        .plan_original_media_semantics(&host)
        .map_err(Error::backend)?
        .compile(&pool)
        .map_err(Error::backend)?;
    let funding = pool
        .prepare_workspace_metadata(
            session.inference_execution_identity(),
            eredu_core::MemoryLimits::unlimited(pool.topology()),
        )
        .map_err(Error::backend)?;
    let binding = session
        .prepare_original_media_semantic_binding(&funding)
        .map_err(Error::backend)?;
    let semantics = A::bind_original_media_semantics(
        admission,
        semantics,
        &sources.inference_blueprint(),
        &host,
        binding,
    )
    .map_err(Error::backend)?;
    let request =
        crate::memory_fixture::request_in(&pool, session.inference_execution_identity(), geometry)
            .map_err(Error::backend)?;
    let metadata = WorkspaceContext::new_with_metadata_funding(MetadataFacts, funding)
        .map_err(Error::backend)?;
    let lowered = eredu_architectures::processor_execution::lower_original_prepared_host_input::<
        _,
        std::convert::Infallible,
    >(&host, &mut NumericProcessorMechanisms::default())
    .map_err(Error::backend)?;
    let plan = A::prepare_bound_original_ingress_plan_with_metadata(
        lowered, semantics, geometry, &metadata,
    )
    .map_err(|e| Error::backend(format!("bound ingress constructor: {e}")))?;
    session
        .prepare_media_prefill_with_metadata(plan, request, &metadata)
        .map_err(|e| Error::backend(format!("retained media constructor: {e}")))
}
