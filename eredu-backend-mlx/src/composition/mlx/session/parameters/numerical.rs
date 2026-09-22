//! Source-qualified resident parameter contractions through the shared worker.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    nn::workspace::{ExistingArrayProjection, ResidentExecutionMechanisms},
    submission_recovery::native_role::physical,
};
use eredu_nn::{
    parameter_values::WorkspaceParameterValues,
    workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataError},
};
use std::mem::{size_of, size_of_val};
#[path = "numerical/effective.rs"]
mod effective;
#[cfg(test)]
pub(super) use effective::InjectedParameterCallbackFailure;
pub(super) use effective::{PreparedOperation, Request};
#[path = "numerical/update.rs"]
mod update;

#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
mod tests {
    use super::*;
    use crate::backend::{
        MlxDeviceIdentity,
        managed_memory::gpu_stream::PreparedExecutionStreams,
        nn::workspace::{MlxCpuMatmulMechanism, MlxMetalWorkspaceMechanisms},
    };
    use eredu_runtime::working_memory::InferenceExecutionIdentity;
    use safemlx::{Device, DeviceType};

    #[test]
    #[ignore = "requires the qualified CPU allocator and matrix kernel"]
    fn resident_projection_preserves_values_result_custody_and_future_admission() {
        if !crate::tests::support::native_process::enter("parameter projection") {
            return;
        }
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let matrix =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, matrix)
            .unwrap()
            .unwrap();
        let backend = MlxBackend::for_prepared_execution_plan(
            streams,
            MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None)
                .unwrap(),
        );
        let environment = backend.original_copy_environment().unwrap();
        let data = (0..24).map(|n| (n as f32 - 7.) * 0.125).collect::<Vec<_>>();
        let array = Array::from_slice(&data, &[2, 3, 4]);
        array.evaluated().unwrap();
        let before = pool.snapshot().unwrap();
        let execution = InferenceExecutionIdentity::default();
        for axis in 0..3 {
            let region = ParameterRegion {
                starts: vec![0, 1, 1],
                shape: vec![2, 2, 3],
            };
            let width = region.shape[axis] as usize;
            let coefficients = (0..2 * width)
                .map(|n| (n as f32 - 2.) * 0.25)
                .collect::<Vec<_>>();
            let projection = ParameterProjection {
                region,
                axis,
                directions: 2,
                coefficients,
            };
            let shape = projection.output_shape(&[2, 3, 4]).unwrap();
            let output = result::PreparedResult::projection(
                &pool,
                "source",
                "weight",
                &shape,
                shape.capacity(),
            )
            .unwrap();
            let funding = pool
                .prepare_workspace_metadata(&execution, pool.configured_limits().clone())
                .unwrap();
            let mechanism = ResidentExecutionMechanisms::from_stream(
                MlxMetalWorkspaceMechanisms::current_host()
                    .unwrap()
                    .original_storage(),
                environment.stream(),
                &funding,
            )
            .unwrap();
            let sources = std::cell::RefCell::new(Vec::new());
            let reader = Visitor {
                parameter: "weight",
                projection: &projection,
                environment: &environment,
                mechanism,
                funding: &funding,
                execution: &execution,
                sources: &sources,
                output: None,
            };
            let values = reader.project(&array).unwrap();
            let mut expected = Vec::new();
            let other = (0..3).filter(|n| *n != axis).collect::<Vec<_>>();
            for a in 0..projection.region.shape[other[0]] as usize {
                for b in 0..projection.region.shape[other[1]] as usize {
                    for direction in 0..2 {
                        let mut sum = 0.;
                        for k in 0..width {
                            let mut index = [0, 1, 1];
                            index[other[0]] += a;
                            index[other[1]] += b;
                            index[axis] += k;
                            sum += data[(index[0] * 3 + index[1]) * 4 + index[2]]
                                * projection.coefficients[direction * width + k];
                        }
                        expected.push(sum);
                    }
                }
            }
            assert_eq!(values.len(), expected.len());
            for (actual, expected) in values.iter().zip(&expected) {
                assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
            }
            let retained = output
                .finish_projection(ParameterProjectionValues {
                    identity: "source".into(),
                    parameter: "weight".into(),
                    source_dtype: InterventionDtype::Float32,
                    shape,
                    values,
                    usage: CaptureUsage::default(),
                })
                .unwrap();
            drop(reader);
            drop(funding);
            let alias = retained.clone();
            assert!(alias.same_storage(&retained));
            drop(retained);
            assert_eq!(alias.values, expected);
            assert_eq!(
                pool.snapshot().unwrap().unquoted_owners,
                before.unquoted_owners
            );
            // This is the same cold-admission gate used by the next request.
            let next = pool
                .prepare_workspace_metadata(&execution, pool.configured_limits().clone())
                .unwrap();
            drop((next, alias));
        }
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires the qualified native factory and source readers"]
    fn prepared_parameter_queries_preserve_three_residencies_and_following_generation() {
        parameter_queries_with_following_generation(0..3, false);
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires the qualified native factory and source readers"]
    fn canonical_parameter_source_survives_cache_eviction_and_final_alias_retirement() {
        parameter_queries_with_following_generation(2..3, true);
    }
    #[cfg(feature = "metal")]
    fn parameter_queries_with_following_generation(
        residencies: std::ops::Range<usize>,
        prepared: bool,
    ) {
        use crate::composition::mlx::session::model_session::{
            disk_layerwise_tests, text_quote::PreparedResidencyFixture,
        };
        use eredu_core::TextGeneration;
        use eredu_runtime::parameter_operations::PreparedParameterLocation;
        if !crate::tests::support::native_process::enter("prepared parameter source") {
            return;
        }
        let fixture = PreparedResidencyFixture::new();
        let limits = CaptureUsage {
            captures: u64::MAX,
            retained_bytes: u64::MAX,
            host_bytes: u64::MAX,
            encoded_bytes: u64::MAX,
        };
        let mut reference = None;
        for residency in residencies {
            let host_before =
                safemlx::host_transfer_memory_stats(safemlx::HostTransferStorageKind::MetalShared)
                    .unwrap();
            let (mut runtime, _artifact, _) = fixture.load(residency, Some(128));
            let id = runtime
                .session()
                .payload
                .model
                .erased()
                .prepared_parameter_slots()
                .iter()
                .find(|slot| {
                    matches!(
                        slot.location,
                        PreparedParameterLocation::Unit { ordinal: 0, .. }
                    ) && slot.materialized.shape.len() == 2
                        && slot.materialized.shape.iter().all(|&n| n >= 2)
                })
                .expect("nonzero matrix in the first retained source unit")
                .parameter
                .id
                .as_str()
                .to_owned();
            let discovery = MlxBackend::parameter_discovery(&mut runtime).unwrap();
            let region = ParameterRegion {
                starts: vec![0, 0],
                shape: vec![2, 2],
            };
            let queried = MlxBackend::query_parameter(
                &mut runtime,
                &discovery.identity,
                &id,
                region.clone(),
                limits,
            )
            .unwrap();
            assert!(queried.values.iter().any(|value| *value != 0.0));
            if let Some(expected) = &reference {
                assert_eq!(&queried.values, expected);
            } else {
                reference = Some(queried.values.clone());
            }
            let alias = queried.clone();
            let projected = MlxBackend::project_parameter(
                &mut runtime,
                &discovery.identity,
                &id,
                ParameterProjection {
                    region,
                    axis: 1,
                    directions: 2,
                    coefficients: vec![1.0, 0.25, -0.5, 0.75],
                },
                limits,
            )
            .unwrap();
            for row in 0..2 {
                let a = queried.values[row * 2];
                let b = queried.values[row * 2 + 1];
                assert!((projected.values[row * 2] - (a + 0.25 * b)).abs() < 1e-5);
                assert!((projected.values[row * 2 + 1] - (-0.5 * a + 0.75 * b)).abs() < 1e-5);
            }
            assert_eq!(fixture.pool.unquoted_owner_count().unwrap(), 0);
            if residency == 2 {
                let (backend, session) = runtime.parts_mut();
                let environment = backend.original_copy_environment().unwrap();
                let (_, layouts) = session.parameter_facts_and_layouts().unwrap();
                let region = ParameterRegion {
                    starts: vec![0, 0],
                    shape: vec![2, 2],
                };
                for _ in 0..2 {
                    let effective = session
                        .read_resident_effective_parameter(
                            &id,
                            &layouts[&id],
                            Request::Read(&region),
                            &environment,
                        )
                        .unwrap()
                        .unwrap();
                    assert_eq!(effective, queried.values);
                }
            }
            let cached = (residency == 2).then(|| {
                let workspace = runtime
                    .session()
                    .payload
                    .model
                    .layerwise_workspace()
                    .unwrap()
                    .unwrap();
                let alias = workspace.test_evict_completed_parameter_alias();
                let allocation = alias.array().allocation_info().unwrap().unwrap();
                assert!(allocation.bytes() > 0);
                let inspection = alias.completed_numerical_source().map(|source| {
                    let inspection = source.budget().clone();
                    assert_eq!(
                        inspection
                            .inspect_array(alias.array())
                            .unwrap()
                            .unwrap()
                            .allocation(),
                        allocation
                    );
                    assert!(inspection.occupied_bytes() >= allocation.bytes());
                    inspection
                });
                if inspection.is_none() {
                    // Metal's same-dtype vector copy can donate the shared Host
                    // backing. Its authenticated receipt remains the source.
                    let host = alias.test_host_source().unwrap();
                    assert_eq!(host.try_allocation_info().unwrap(), allocation);
                    assert!(
                        host.attachment_receipt()
                            .unwrap()
                            .matches(allocation, &fixture.pool)
                    );
                }
                (alias, inspection, allocation)
            });
            let config = disk_layerwise_tests::config(0.0, 1, u64::MAX);
            let output = if prepared {
                let request = eredu_core::GenerationSequenceRequest::new(
                    config.sampling().max_new_tokens.unwrap(),
                    &[],
                );
                TextGeneration::from_token_ids_with_sequence(
                    &mut runtime,
                    eredu_core::TokenIdsInputPlan::new(&[1, 2]).unwrap(),
                    config,
                    eredu_core::TokenFilter::All,
                    None,
                    request,
                )
            } else {
                TextGeneration::new(&mut runtime, vec![1, 2], config)
            }
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
            assert!(!output.is_empty());
            drop(queried);
            assert_eq!(&alias.values, reference.as_ref().unwrap());
            assert!(projected.values.iter().all(|value| value.is_finite()));
            drop((output, alias, projected, discovery, runtime));
            if let Some((alias, inspection, allocation)) = cached {
                disk_layerwise_tests::reclaim();
                assert_eq!(alias.array().allocation_info().unwrap(), Some(allocation));
                if let Some(inspection) = &inspection {
                    assert_eq!(
                        inspection
                            .inspect_array(alias.array())
                            .unwrap()
                            .unwrap()
                            .allocation(),
                        allocation
                    );
                    assert!(inspection.occupied_bytes() >= allocation.bytes());
                } else {
                    let host = alias.test_host_source().unwrap();
                    assert!(
                        host.attachment_receipt()
                            .unwrap()
                            .matches(allocation, &fixture.pool)
                    );
                    let live = safemlx::host_transfer_memory_stats(
                        safemlx::HostTransferStorageKind::MetalShared,
                    )
                    .unwrap();
                    assert!(
                        live.active_bytes
                            >= host_before
                                .active_bytes
                                .checked_add(allocation.bytes())
                                .unwrap()
                    );
                    assert!(live.active_allocations > host_before.active_allocations);
                }
                drop(alias);
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    disk_layerwise_tests::reclaim();
                    let live = safemlx::host_transfer_memory_stats(
                        safemlx::HostTransferStorageKind::MetalShared,
                    )
                    .unwrap();
                    inspection
                        .as_ref()
                        .is_none_or(|inspection| inspection.occupied_bytes() == 0)
                        && live.active_bytes == host_before.active_bytes
                        && live.active_allocations == host_before.active_allocations
                });
                drop(inspection);
            }
            crate::backend::submission_recovery::wait_for_retirement(|| {
                disk_layerwise_tests::reclaim();
                let live = fixture.pool.snapshot().unwrap();
                live.reservations == 0 && live.unquoted_owners == 0
            });
        }
    }
}

struct Visitor<'a> {
    parameter: &'a str,
    projection: &'a ParameterProjection,
    environment: &'a OriginalCopyEnvironment<'a>,
    mechanism: ResidentExecutionMechanisms,
    funding: &'a HostMetadataFunding,
    execution: &'a eredu_runtime::working_memory::InferenceExecutionIdentity,
    sources: &'a std::cell::RefCell<Vec<crate::backend::nn::workspace::CompletedParameterSource>>,
    output: Option<Result<Vec<f32>, Error>>,
}

fn coefficient_source(
    shape: &[i32],
    context: &WorkspaceContext,
) -> Result<eredu_nn::workspace::WorkspaceTensor, eredu_nn::Error> {
    crate::backend::nn::workspace::host_array::trace(shape, Dtype::Float32, context)
}

fn missing() -> Error {
    Error::OriginalSourceContract {
        stage: "parameter numerical control quotation",
        cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
    }
}

impl ParameterSlotVisitor<MlxTensor> for Visitor<'_> {
    fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &MlxTensor) {
        if metadata.id().as_str() != self.parameter {
            return;
        }
        if self.output.is_some() {
            self.output = Some(Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )));
            return;
        }
        let array = value.as_array();
        if !matches!(
            array.dtype(),
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ) {
            return;
        }
        self.output = Some(self.project(array));
    }
}

impl Visitor<'_> {
    fn project(&self, array: &Array) -> Result<Vec<f32>, Error> {
        let context = self
            .mechanism
            .context(self.funding.clone())
            .map_err(|cause| Error::Neural(cause.into()))?;
        let mut sources = ExistingArrayProjection::with_source_count(&context, 1)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let source = sources.project(array).map_err(Error::Neural)?;
        if !sources.is_complete() {
            return Err(missing());
        }
        context.begin_span();
        eredu_nn::parameter_values::project_parameter(
            WorkspaceParameterValues::projection_with_coefficients(
                &source,
                self.projection,
                &context,
                coefficient_source,
            )
            .map_err(Error::Neural)?,
        )
        .map_err(Error::Neural)?;
        let report = context.finish_report(&[source]).map_err(Error::Neural)?;
        execute(
            &report,
            &[array],
            self.mechanism,
            &context,
            self.environment,
            self.execution,
            self.sources,
            encoding::projection_host_control_bytes(self.projection).ok_or_else(missing)?,
            || encoding::project_effective(array, self.projection, self.environment.stream()),
        )
    }
}

fn execute<T>(
    report: &eredu_nn::workspace::WorkspaceTraceReport,
    inputs: &[&Array],
    mechanism: ResidentExecutionMechanisms,
    context: &WorkspaceContext,
    environment: &OriginalCopyEnvironment<'_>,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    sources: &std::cell::RefCell<Vec<crate::backend::nn::workspace::CompletedParameterSource>>,
    host_controls: usize,
    operation: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    let completed = physical::execute_numerical(
        report,
        inputs,
        inputs.len(),
        mechanism,
        context,
        environment,
        execution,
        host_controls,
        operation,
    )?;
    let mut sources = sources
        .try_borrow_mut()
        .map_err(|_| Error::PrefillScopeReentrant)?;
    context
        .reserve_metadata_vec(&mut sources, 1)
        .map_err(Error::Neural)?;
    sources.push(completed.source.into());
    Ok(completed.value)
}

impl MlxModelSession {
    /// Uses only an already resident immutable slot. A miss leaves materialization
    /// with its existing selected residency source; no replacement token source
    /// is inferred from this numerical plan.
    pub(super) fn project_resident_parameter(
        &mut self,
        parameter: &str,
        projection: &ParameterProjection,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Option<Vec<f32>>, Error> {
        let model = self
            .original_model_source()
            .map_err(Error::PrefillControl)?;
        let mechanism = model.resident_workspace_mechanisms().ok_or_else(missing)?;
        let pool = environment.pool();
        if !pool.same_ledger(&self.payload.memory_ledger) {
            return Err(missing());
        }
        let funding = pool
            .prepare_workspace_metadata(
                model.erased().inference_execution_identity(),
                pool.configured_limits().clone(),
            )
            .map_err(Error::WorkspacePlanning)?;
        let frames = [
            size_of::<Visitor<'_>>(),
            size_of::<eredu_runtime::working_memory::InferenceExecutionIdentity>(),
            size_of::<
                std::cell::RefCell<Vec<crate::backend::nn::workspace::CompletedParameterSource>>,
            >(),
            size_of::<Option<Result<Vec<f32>, Error>>>(),
            size_of::<Result<Option<Vec<f32>>, Error>>(),
            OriginalCopyEnvironment::control_bytes().ok_or_else(missing)?,
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(eredu_core::HostMetadataFundingError::Overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        self.with_model_operation_funded(funding.clone(), |model| {
            let sources = std::cell::RefCell::new(Vec::new());
            let execution = model.erased().inference_execution_identity().clone();
            let mut visitor = Visitor {
                parameter,
                projection,
                environment,
                mechanism,
                funding: &funding,
                execution: &execution,
                sources: &sources,
                output: None,
            };
            model.erased_mut().visit_loaded_parameters(&mut visitor);
            visitor.output.transpose()
        })
    }
}

#[cfg(test)]
pub(super) fn update_for_test(
    source: &Array,
    region: &ParameterRegion,
    edit: &ParameterUpdate,
    stream: &Stream,
) -> Result<Array, Error> {
    eredu_nn::parameter_values::update_parameter(
        update::NativeUpdate::new(region, edit, stream)?,
        source,
        edit,
    )?
    .ok_or_else(missing)
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
#[path = "numerical/overlay_tests.rs"]
mod overlay_tests;
