//! Actual resident primary/companion loans through the same numerical role.
use super::*;
use eredu_nn::parameter_values::{ParameterDecoding, WorkspaceParameterDecoding};

#[cfg(test)]
#[derive(Debug, thiserror::Error)]
#[error("injected parameter callback rejection")]
pub(in super::super) struct InjectedParameterCallbackFailure;

pub(in super::super) enum Request<'a> {
    Read(&'a ParameterRegion),
    Project(&'a ParameterProjection),
}
impl Request<'_> {
    fn trace(
        &self,
        value: &eredu_nn::workspace::WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        match self {
            Self::Read(region) => eredu_nn::parameter_values::read_parameter(
                WorkspaceParameterValues::read(value, region, context).map_err(Error::Neural)?,
            ),
            Self::Project(projection) => eredu_nn::parameter_values::project_parameter(
                WorkspaceParameterValues::projection_with_coefficients(
                    value,
                    projection,
                    context,
                    coefficient_source,
                )
                .map_err(Error::Neural)?,
            ),
        }
        .map_err(Error::Neural)
    }
    fn execute(&self, value: &Array, stream: &Stream) -> Result<Vec<f32>, Error> {
        match self {
            Self::Read(region) => encoding::read_effective(value, region, stream),
            Self::Project(projection) => encoding::project_effective(value, projection, stream),
        }
    }
    fn host_bytes(&self) -> Option<usize> {
        match self {
            Self::Read(region) => encoding::read_host_control_bytes(region),
            Self::Project(projection) => encoding::projection_host_control_bytes(projection),
        }
    }
}

/// Every final C handle is allocated only after its exact constructor charge.
/// Handles retain existing descriptors; no graph or numerical backing is born.
struct Sources<'a> {
    ids: [Option<&'a str>; 3],
    values: [Option<Array>; 3],
    slots: [Option<safemlx::PreparedArrayClone>; 3],
    error: Option<Error>,
    context: &'a WorkspaceContext,
}
impl<'a> Sources<'a> {
    fn new(ids: [Option<&'a str>; 3], context: &'a WorkspaceContext) -> Result<Self, Error> {
        let handle = Array::inspection_clone_handle_bytes()
            .checked_add(safemlx::PreparedArrayClone::control_bytes().ok_or_else(missing)?)
            .ok_or_else(missing)?;
        let bytes = handle
            .checked_mul(ids.iter().flatten().count())
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(size_of::<Result<Self, Error>>()))
            .and_then(|n| {
                n.checked_add(WorkspaceContext::metadata_source_bytes::<
                    safemlx::PreparedArrayCloneCause,
                >()?)
            })
            .ok_or_else(missing)?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| Error::Neural(cause.into()))?;
        let mut value = Self {
            ids,
            values: [None, None, None],
            slots: [None, None, None],
            error: None,
            context,
        };
        for (id, slot) in value.ids.iter().zip(&mut value.slots) {
            if id.is_some() {
                *slot = Some(
                    safemlx::PreparedArrayClone::try_prepare_for_inspection()
                        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?,
                );
            }
        }
        Ok(value)
    }
}
impl ParameterSlotVisitor<MlxTensor> for Sources<'_> {
    fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &MlxTensor) {
        if self.error.is_some() {
            return;
        }
        for index in 0..self.ids.len() {
            if self.ids[index] != Some(metadata.id().as_str()) {
                continue;
            }
            if self.values[index].is_some() {
                self.error = Some(Error::OriginalSourceContract {
                    stage: "effective parameter duplicate source slot",
                    cause: eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                });
                return;
            }
            match self.slots[index]
                .as_mut()
                .expect("prepared selected slot")
                .fill_for_inspection(value.as_array())
            {
                Ok(value) => self.values[index] = Some(value),
                Err(cause) => self.error = Some(Error::Neural(self.context.metadata_source(cause))),
            }
        }
    }
}

/// One actual idle parameter operation and its independently retained cold payer.
pub(in super::super) struct PreparedOperation<'a> {
    pub(in super::super) environment: &'a OriginalCopyEnvironment<'a>,
    pub(in super::super) sources:
        std::cell::RefCell<Vec<crate::backend::nn::workspace::CompletedParameterSource>>,
    pub(in super::super) execution: eredu_runtime::working_memory::InferenceExecutionIdentity,
    pub(in super::super) mechanism: ResidentExecutionMechanisms,
    pub(in super::super) context: WorkspaceContext,
    pub(in super::super) funding: HostMetadataFunding,
}
impl<'a> PreparedOperation<'a> {
    pub(in super::super) fn new(
        session: &MlxModelSession,
        environment: &'a OriginalCopyEnvironment<'a>,
    ) -> Result<Self, Error> {
        let model = session
            .original_model_source()
            .map_err(Error::PrefillControl)?;
        let mechanism =
            model
                .resident_workspace_mechanisms()
                .ok_or(Error::OriginalSourceContract {
                    stage: "parameter operation resident mechanisms",
                    cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                })?;
        let pool = environment.pool();
        if !pool.same_ledger(&session.payload.memory_ledger) {
            return Err(Error::OriginalSourceContract {
                stage: "parameter operation ledger identity",
                cause: eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            });
        }
        let funding = pool
            .prepare_workspace_metadata(
                model.erased().inference_execution_identity(),
                pool.configured_limits().clone(),
            )
            .map_err(Error::WorkspacePlanning)?;
        let context = mechanism
            .context(funding.clone())
            .map_err(|cause| Error::Neural(cause.into()))?;
        context
            .charge_metadata(size_of::<(Self, Result<Self, Error>)>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        let mut sources = context
            .metadata_vec(model.parameter_sources().prepared_sources().count())
            .map_err(Error::Neural)?;
        sources.extend(model.parameter_sources().prepared_sources());
        Ok(Self {
            environment,
            funding,
            context,
            execution: model.erased().inference_execution_identity().clone(),
            mechanism,
            sources: std::cell::RefCell::new(sources),
        })
    }
    pub(in super::super) fn preparation<'b>(
        &'b self,
        parameters: &'b [Option<&'b str>],
    ) -> crate::backend::runtime::execution::generic::MlxParameterPreparation<'b> {
        crate::backend::runtime::execution::generic::MlxParameterPreparation {
            environment: self.environment,
            funding: &self.funding,
            execution: &self.execution,
            parameters,
            mechanism: self.mechanism,
            sources: &self.sources,
        }
    }
    pub(in super::super) fn take_sources(
        &self,
    ) -> Vec<crate::backend::nn::workspace::CompletedParameterSource> {
        std::mem::take(&mut *self.sources.borrow_mut())
    }
}
impl MlxModelSession {
    pub(in super::super) fn read_resident_effective_parameter(
        &mut self,
        parameter: &str,
        layout: &EffectiveLayout,
        request: Request<'_>,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Option<Vec<f32>>, Error> {
        let prepared = PreparedOperation::new(self, environment)?;
        self.with_parameter_sources(parameter, layout, &prepared, |sources| {
            execute_sources(
                sources,
                layout,
                &request,
                prepared.mechanism,
                &prepared.context,
                environment,
                &prepared.execution,
                &prepared.sources,
            )
        })
        .map(Some)
    }
    fn with_parameter_sources<T>(
        &mut self,
        parameter: &str,
        layout: &EffectiveLayout,
        prepared: &PreparedOperation<'_>,
        mut operation: impl FnMut(&mut Sources<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let PreparedOperation {
            environment,
            funding,
            context,
            mechanism,
            ..
        } = prepared;
        let mechanism = *mechanism;
        context
            .charge_metadata(size_of::<(
                T,
                Result<T, Error>,
                Option<T>,
                crate::backend::runtime::execution::generic::MlxParameterPreparation<'_>,
                Option<&crate::backend::runtime::execution::generic::MlxParameterPreparation<'_>>,
                [&Array; 3],
                [Option<eredu_nn::workspace::WorkspaceTensor>; 3],
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        self.with_model_inspection_funded(funding.clone(), |model| {
            let mut sources = Sources::new(layout.source_ids(parameter), context)?;
            model.erased_mut().visit_loaded_parameters(&mut sources);
            if let Some(cause) = sources.error.take() {
                return Err(cause);
            }
            if sources
                .ids
                .iter()
                .zip(&sources.values)
                .all(|(id, value)| id.is_none() || value.is_some())
            {
                return operation(&mut sources);
            }
            use eredu_runtime::parameter_operations::PreparedParameterLocation;
            let mut owner = None;
            for id in sources.ids.iter().flatten() {
                let row = model
                    .erased()
                    .prepared_parameter_slots()
                    .iter()
                    .find(|row| row.parameter.id.as_str() == *id)
                    .ok_or(Error::OriginalSourceContract {
                        stage: "parameter operation prepared source slot",
                        cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                    })?;
                if owner.as_ref().is_some_and(|previous| {
                    !matches!(
                        (previous, &row.location),
                        (
                            PreparedParameterLocation::Static { .. },
                            PreparedParameterLocation::Static { .. }
                        )
                    ) && previous != &row.location
                }) {
                    return Err(Error::OriginalSourceContract {
                        stage: "effective parameter companion owner",
                        cause: eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                    });
                }
                if owner.is_none() {
                    if let PreparedParameterLocation::Static { role } = &row.location {
                        context
                            .charge_metadata(role.len())
                            .map_err(|cause| Error::Neural(cause.into()))?;
                    }
                    owner = Some(row.location.clone());
                }
            }
            let owner = owner.ok_or_else(missing)?;
            let execution = model.erased().inference_execution_identity().clone();
            let preparation =
                crate::backend::runtime::execution::generic::MlxParameterPreparation {
                    environment,
                    funding: &funding,
                    execution: &execution,
                    parameters: &sources.ids,
                    mechanism,
                    sources: &prepared.sources,
                };
            // Unit/static owners visit their actual module. Selection stays in
            // the fixed borrowed source IDs; their adapter uses no bank member set.
            let selected = std::collections::BTreeSet::new();
            let mut result = None;
            let available = model
                .erased_mut()
                .with_parameter_slots(
                    &owner,
                    &selected,
                    &mut |visit| {
                        let mut sources = Sources::new(layout.source_ids(parameter), context)?;
                        visit(&mut sources);
                        if let Some(cause) = sources.error.take() {
                            return Err(cause);
                        }
                        result = Some(operation(&mut sources)?);
                        Ok(())
                    },
                    environment.stream(),
                    Some(&preparation),
                )
                .map_err(|cause| match cause {
                    Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                    ) => Error::OriginalSourceContract {
                        stage: "parameter operation selected source preparation",
                        cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                    },
                    cause => cause,
                })?;
            if !available {
                return Err(Error::OriginalSourceContract {
                    stage: "parameter operation selected source owner",
                    cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                });
            }
            result.ok_or(Error::OriginalSourceContract {
                stage: "parameter operation selected source callback",
                cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            })
        })
    }
}

fn execute_sources(
    sources: &Sources<'_>,
    layout: &EffectiveLayout,
    request: &Request<'_>,
    mechanism: ResidentExecutionMechanisms,
    context: &WorkspaceContext,
    environment: &OriginalCopyEnvironment<'_>,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    retained_sources: &std::cell::RefCell<
        Vec<crate::backend::nn::workspace::CompletedParameterSource>,
    >,
) -> Result<Vec<f32>, Error> {
    execute_decoded(
        sources,
        layout,
        mechanism,
        context,
        environment,
        execution,
        retained_sources,
        request.host_bytes().ok_or_else(missing)?,
        |value| {
            request.trace(&value, context)?;
            Ok(None)
        },
        |value| request.execute(&value, environment.stream()),
    )
}

fn execute_decoded<T>(
    sources: &Sources<'_>,
    layout: &EffectiveLayout,
    mechanism: ResidentExecutionMechanisms,
    context: &WorkspaceContext,
    environment: &OriginalCopyEnvironment<'_>,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    retained_sources: &std::cell::RefCell<
        Vec<crate::backend::nn::workspace::CompletedParameterSource>,
    >,
    host_controls: usize,
    trace: impl FnOnce(
        eredu_nn::workspace::WorkspaceTensor,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceTensor>, Error>,
    operation: impl FnOnce(Array) -> Result<T, Error>,
) -> Result<T, Error> {
    let source_failure = |stage| Error::OriginalSourceContract {
        stage,
        cause: eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
    };
    let weight = sources.values[0]
        .as_ref()
        .ok_or_else(|| source_failure("effective parameter primary source"))?;
    if sources
        .ids
        .iter()
        .zip(&sources.values)
        .any(|(id, value)| id.is_some() && value.is_none())
    {
        return Err(source_failure("effective parameter companion source"));
    }
    let mut projection = ExistingArrayProjection::with_source_count(
        context,
        sources.values.iter().flatten().count(),
    )
    .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    let mut values = [None, None, None];
    for (value, projected) in sources.values.iter().zip(&mut values) {
        if let Some(value) = value {
            *projected = Some(projection.project(value).map_err(Error::Neural)?);
        }
    }
    if !projection.is_complete() {
        return Err(source_failure(
            "effective parameter completed source projection",
        ));
    }
    let mut shape = context
        .metadata_vec(layout.shape.len())
        .map_err(Error::Neural)?;
    for &extent in &layout.shape {
        shape.push(
            i32::try_from(extent)
                .map_err(|_| Error::Neural(WorkspaceMetadataError::Overflow.into()))?,
        );
    }
    context.begin_span();
    let effective = eredu_nn::parameter_values::decode_parameter(
        WorkspaceParameterDecoding::new(
            values[0].as_ref().expect("primary source"),
            values[1].as_ref(),
            values[2].as_ref(),
            &shape,
            context,
        )
        .map_err(Error::Neural)?,
        ParameterDecoding {
            format: layout.format,
            row_layout: layout.row_layout,
        },
    )
    .map_err(Error::Neural)?;
    let output = trace(effective)?;
    let mut retained = context
        .metadata_vec(values.len() + 1)
        .map_err(Error::Neural)?;
    retained.extend(values.into_iter().flatten());
    retained.extend(output);
    let report = context.finish_report(&retained).map_err(Error::Neural)?;
    let mut inputs = context
        .metadata_vec(sources.values.len())
        .map_err(Error::Neural)?;
    inputs.extend(sources.values.iter().flatten());
    let host = host_controls
        .checked_add(encoding::decoding_host_control_bytes(layout).ok_or_else(missing)?)
        .ok_or_else(missing)?;
    execute(
        &report,
        &inputs,
        mechanism,
        context,
        environment,
        execution,
        retained_sources,
        host,
        || {
            let effective = layout.decode_sources(
                weight,
                sources.values[1].as_ref(),
                sources.values[2].as_ref(),
                environment.stream(),
            )?;
            operation(effective)
        },
    )
}

impl MlxModelSession {
    /// Produces complete replacement and original roots through the same source
    /// loan and numerical reducer used by reads and projections.
    pub(in super::super) fn prepare_parameter_update<'a>(
        &mut self,
        parameter: &str,
        layout: &EffectiveLayout,
        edits: &[(&'a ParameterRegion, &'a ParameterUpdate)],
        prepared: &PreparedOperation<'_>,
    ) -> Result<(MlxTensor, MlxTensor), Error> {
        self.with_parameter_sources(parameter, layout, prepared, |sources| {
            let context = &prepared.context;
            let mut host = edits
                .iter()
                .try_fold(0usize, |total, (region, _)| {
                    total.checked_add(update::NativeUpdate::control_bytes(region)?)
                })
                .ok_or_else(missing)?;
            if edits.is_empty() {
                host = host
                    .checked_add(
                        crate::backend::runtime::cache::completed_borrow_control_bytes()
                            .ok_or_else(missing)?,
                    )
                    .ok_or_else(missing)?;
            }
            let replacement = execute_decoded(
                sources,
                layout,
                prepared.mechanism,
                context,
                prepared.environment,
                &prepared.execution,
                &prepared.sources,
                host,
                |mut value| {
                    for (region, update) in edits {
                        value = eredu_nn::parameter_values::update_parameter(
                            eredu_nn::parameter_values::WorkspaceParameterUpdate::new(
                                region,
                                context,
                                coefficient_source,
                            )
                            .map_err(Error::Neural)?,
                            &value,
                            update,
                        )
                        .map_err(Error::Neural)?
                        .ok_or_else(missing)?;
                    }
                    if edits.is_empty() {
                        context.complete_values(&[&value]).map_err(Error::Neural)?;
                    }
                    Ok(Some(value))
                },
                |mut value| {
                    for (region, edit) in edits {
                        value = eredu_nn::parameter_values::update_parameter(
                            update::NativeUpdate::new(region, edit, prepared.environment.stream())?,
                            &value,
                            edit,
                        )?
                        .ok_or_else(|| {
                            match context.metadata_string(format_args!(
                                "parameter edit overflows target dtype"
                            )) {
                                Ok(message) => Error::Neural(
                                    context.metadata_source(ParameterError::Invalid(message)),
                                ),
                                Err(cause) => Error::Neural(cause),
                            }
                        })?;
                    }
                    if edits.is_empty() {
                        crate::backend::runtime::cache::complete_and_borrow(
                            &value,
                            prepared.environment.stream(),
                        )?;
                    }
                    Ok(value)
                },
            )?;
            // A selected residency loan can own its temporary primary in a
            // source arena. Restoration retains an independent completed root
            // with its own exact numerical custody after that loan retires.
            let copy_context = prepared
                .mechanism
                .context(prepared.funding.clone())
                .map_err(|cause| Error::Neural(cause.into()))?;
            let original = physical::copy_array(
                sources.values[0].as_ref().ok_or_else(missing)?,
                prepared.mechanism,
                &copy_context,
                prepared.environment,
                &prepared.execution,
                prepared.environment.stream(),
            )?;
            prepared
                .preparation(&sources.ids)
                .retain_source(original.source, &copy_context)?;
            Ok((
                MlxTensor::from_array(original.value),
                MlxTensor::from_array(replacement),
            ))
        })
    }
}

#[cfg(test)]
impl MlxModelSession {
    pub(in super::super) fn reject_parameter_callback_for_test(
        &mut self,
        parameter: &str,
        layout: &EffectiveLayout,
        region: &ParameterRegion,
        prepared: &PreparedOperation<'_>,
    ) -> Result<(), Error> {
        self.with_parameter_sources(parameter, layout, prepared, |sources| {
            let values = execute_sources(
                sources,
                layout,
                &Request::Read(region),
                prepared.mechanism,
                &prepared.context,
                prepared.environment,
                &prepared.execution,
                &prepared.sources,
            )?;
            assert!(!values.is_empty());
            Err(Error::Neural(
                prepared
                    .context
                    .metadata_source(InjectedParameterCallbackFailure),
            ))
        })
    }
}
