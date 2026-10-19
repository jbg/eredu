//! Target-bound neural geometry and placement retained before source construction.
use super::*;

/// Source placement and successful invocation admission share one immutable
/// target selection. Callers never supply another target during admission.
pub(super) enum AssistantTargetSelection {
    Placed(crate::SelectedPreparation),
    Admitted(AdmittedAssistantTarget),
}

#[derive(Clone)]
pub(super) struct AdmittedAssistantTarget {
    pub(super) selected: crate::SelectedPreparation,
    proposals: NonZeroUsize,
    invocations: AdmittedTargetInvocations,
}

/// Only complete capture/static admission grants a coordinated target contract.
#[derive(Clone, Copy)]
enum AdmittedTargetInvocations {
    Local(NonZeroUsize),
    Coordinated(NonZeroUsize),
}

impl AdmittedTargetInvocations {
    fn capture_sequence(self) -> NonZeroUsize {
        match self {
            Self::Local(sequence) | Self::Coordinated(sequence) => sequence,
        }
    }
}

impl AssistantTargetSelection {
    fn selected(&self) -> &crate::SelectedPreparation {
        match self {
            Self::Placed(selected) => selected,
            Self::Admitted(admitted) => &admitted.selected,
        }
    }

    pub(super) fn validate_request(
        &self,
        request: &ExternalSpeculativeContractRequest,
        profile: &ExternalAssistantTargetProfile,
    ) -> Result<(), ArtifactError> {
        let Self::Admitted(admitted) = self else {
            return Err(invalid_assistant(
                "external speculative target invocations are not admitted",
            ));
        };
        let rank = target_rank(&admitted.selected, profile, None)?;
        if rank != request.topology {
            return Err(invalid_assistant(
                "external speculative topology differs from admitted target invocations",
            ));
        }
        if admitted.proposals != request.maximum_draft_tokens {
            return Err(invalid_assistant(
                "external speculative capacity differs from admitted target invocations",
            ));
        }
        let sequence = profile
            .maximum_capture_sequence_length()?
            .min(request.maximum_capture_sequence_length);
        if sequence > admitted.invocations.capture_sequence() {
            return Err(invalid_assistant(
                "external speculative capture exceeds admitted target invocations",
            ));
        }
        if request.topology.world_size() != 1
            && !matches!(
                admitted.invocations,
                AdmittedTargetInvocations::Coordinated(_)
            )
        {
            return Err(invalid_assistant(
                "partitioned external speculation requires coordinated target capture and static admission",
            ));
        }
        Ok(())
    }
}

pub(super) struct NormalizedExternalAssistant<A: ExternalAssistantArchitecture> {
    pub(super) selected: SelectedExternalAssistant<A>,
    source_module_config: A::ModuleConfig,
    module_config: A::ModuleConfig,
    parameter_layout: Option<eredu_runtime::LocalModelLayout>,
}

impl<A: ExternalAssistantArchitecture> NormalizedExternalAssistant<A> {
    pub(super) fn new(
        selected: SelectedExternalAssistant<A>,
        target: Option<&ExternalAssistantTargetProfile>,
    ) -> Result<Self, String> {
        let source_module_config = A::module_config(&selected.source_config, target)?;
        let module_config = A::module_config(&selected.config, target)?;
        Ok(Self {
            selected,
            source_module_config,
            module_config,
            parameter_layout: None,
        })
    }

    pub(super) fn prepare_source(
        self,
        max_cached_sources: usize,
    ) -> Result<PreparedExternalAssistantSource<A>, ArtifactError> {
        self.selected.prepare_source(
            max_cached_sources,
            self.source_module_config,
            self.module_config,
            self.parameter_layout,
        )
    }
}

pub(super) fn partition_muse_construction(
    source: &MuseAssistantConstruction,
    selected: &MuseAssistantConstruction,
    tasks: &[ReplicatedTextMaterializationTask],
    rank: ParallelRankTopology,
) -> Result<
    (
        MuseAssistantConstruction,
        MuseAssistantConstruction,
        eredu_runtime::LocalModelLayout,
    ),
    ArtifactError,
> {
    let (selected, parameters) = selected
        .partition_tensor_parallel(rank)
        .map_err(invalid_assistant)?;
    let selected = selected.with_materialization_precisions(tasks);
    let (_, layout) = parameters
        .partition_layout()
        .ok_or_else(|| invalid_assistant("assistant partition omitted its exact layout"))?;
    let source = source
        .source_for_selected(&selected)
        .map_err(invalid_assistant)?;
    Ok((source, selected, layout.clone()))
}

pub(super) fn target_rank(
    target: &crate::SelectedPreparation,
    expected: &ExternalAssistantTargetProfile,
    placement: Option<ExternalAssistantSourcePlacement>,
) -> Result<ParallelRankTopology, ArtifactError> {
    let profile = target
        .retained_architecture()
        .and_then(|architecture| architecture.external_assistant_target_profile())
        .ok_or_else(|| invalid_assistant("target selection omitted its assistant profile"))?;
    if profile.speculative_identity()? != expected.speculative_identity()? {
        return Err(invalid_assistant(
            "assistant source belongs to a different target profile",
        ));
    }
    if placement.is_some() {
        return Err(invalid_assistant(
            "assistant source placement is already selected",
        ));
    }
    Ok(target.execution().parallel_topology().unwrap_or_else(|| {
        ParallelRankTopology::new(eredu_core::ParallelTopology::new(1, 1, 1, 1).unwrap(), 0)
            .unwrap()
    }))
}

pub(super) fn validate_tensor_group(
    target: &crate::SelectedPreparation,
    selected: &MuseAssistantConstruction,
) -> Result<(), ArtifactError> {
    let execution = target.execution();
    let id = execution
        .partitioned_tensor_group()
        .ok_or_else(|| invalid_assistant("target omitted its tensor group"))?;
    let group = execution
        .communication_manifest()
        .and_then(|manifest| manifest.groups().iter().find(|group| group.id() == id))
        .ok_or_else(|| invalid_assistant("target omitted its tensor group manifest"))?;
    let requirements = selected
        .execution_definition()
        .tensor_parallel_requirements()
        .map_err(invalid_assistant)?
        .ok_or_else(|| invalid_assistant("partitioned source omitted its reductions"))?;
    for requirement in requirements.operations() {
        group
            .requirements()
            .validate_requirement(requirement)
            .map_err(invalid_assistant)?;
    }
    Ok(())
}

impl CompatibleExternalAssistantPreparation {
    /// Admits target-owned static calls, verification and captures against the
    /// retained selection without opening payloads or creating native resources.
    /// Later contracts retain this exact proposal capacity and capture bound.
    pub fn admit_target_invocations(
        &mut self,
        proposals: NonZeroUsize,
    ) -> Result<(), ArtifactError> {
        let binding = self.target_selection.as_ref().ok_or_else(|| {
            invalid_assistant("assistant invocation admission requires a retained target selection")
        })?;
        if matches!(binding, AssistantTargetSelection::Admitted(_)) {
            return Err(invalid_assistant(
                "assistant target invocations are already admitted",
            ));
        }
        let target = binding.selected();
        let invocations = self.check_target_invocations(target, proposals)?;
        self.target_selection = Some(AssistantTargetSelection::Admitted(
            AdmittedAssistantTarget {
                selected: target.clone(),
                proposals,
                invocations,
            },
        ));
        Ok(())
    }

    pub(super) fn take_admitted_target(
        &mut self,
    ) -> Result<AdmittedAssistantTarget, ArtifactError> {
        match self.target_selection.take() {
            Some(AssistantTargetSelection::Admitted(admitted)) => Ok(admitted),
            _ => Err(invalid_assistant(
                "assistant target invocations are not admitted",
            )),
        }
    }

    fn check_target_invocations(
        &self,
        target: &crate::SelectedPreparation,
        proposals: NonZeroUsize,
    ) -> Result<AdmittedTargetInvocations, ArtifactError> {
        let rank = target_rank(target, &self.target, None)?;
        let retained_rank = match self.placement {
            Some(ExternalAssistantSourcePlacement::Replicated(rank))
            | Some(ExternalAssistantSourcePlacement::TensorParallel(rank)) => rank,
            None => {
                return Err(invalid_assistant(
                    "assistant target invocation admission requires selected source placement",
                ))
            }
        };
        if retained_rank != rank {
            return Err(invalid_assistant(
                "assistant target invocation rank differs from source placement",
            ));
        }
        let DispatchedNormalizedExternalAssistant::Muse(construction) = &self.preparation else {
            // Gemma's shared-attention compatibility is established separately;
            // its source selector currently admits only a singleton target.
            return self
                .target
                .maximum_capture_sequence_length()
                .map(AdmittedTargetInvocations::Local);
        };
        let invocations = construction
            .module_config
            .execution_definition()
            .target_static_invocations(proposals)
            .map_err(invalid_assistant)?;
        let execution = target.execution();
        if execution.parallel_topology().is_none() {
            return self
                .target
                .maximum_capture_sequence_length()
                .map(AdmittedTargetInvocations::Local);
        }
        let ExternalAssistantTargetProfile::Muse(args) = &self.target else {
            return Err(invalid_assistant(
                "Muse static invocation has no Muse target",
            ));
        };
        let args = muse::with_checkpoint_formats(
            args,
            crate::replicated_text::selected_formats(execution.text_realization()),
        )
        .map_err(invalid_assistant)?;
        let parameters = muse::model::parameter_description(&args).map_err(invalid_assistant)?;
        let index = crate::parameter_partition::ParameterMemberIndex::new(
            execution.text_realization().materialization_tasks(),
            &parameters,
        );
        let description = super::static_placement::TargetStaticDescription::muse(&args);
        for (kind, shape) in invocations {
            let admitted = super::static_placement::PreparedTargetStaticPlacement::prepare_cold(
                execution,
                &parameters,
                &index,
                &description,
                kind,
            )
            .and_then(|placement| {
                placement
                    .ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "partitioned target omitted static source placement",
                        )
                    })?
                    .resolve_admitted(
                        kind,
                        &shape,
                        description.compute_precisions(
                            kind,
                            execution.text_realization().materialization_tasks(),
                        ),
                    )
            });
            admitted.map_err(|error| {
                invalid_assistant(format!(
                    "external assistant target {kind:?} is not admitted: {error}"
                ))
            })?;
        }
        // The shared speculative transaction verifies the pending committed
        // token followed by the selected proposals in one target decode call.
        let verification_rows = proposals.get().checked_add(1).ok_or_else(|| {
            invalid_assistant("external assistant verification row count overflows")
        })?;
        let (maximum_batch, maximum_sequence) = execution
            .partitioned_invocation_limits()
            .ok_or_else(|| invalid_assistant("partitioned target omitted invocation limits"))?;
        if maximum_batch < 1
            || maximum_sequence <= 0
            || verification_rows > maximum_sequence as usize
        {
            return Err(invalid_assistant(format!(
                "external assistant verification requires {verification_rows} rows; selected target admits {maximum_sequence}"
            )));
        }
        let descriptor = target
            .retained_architecture()
            .ok_or_else(|| invalid_assistant("target omitted its retained architecture"))?
            .architecture_descriptor();
        let paths = match &self.capture {
            crate::composite_execution::ExternalPredictionCaptureRequest::Muse {
                target_paths,
                ..
            } => target_paths,
            _ => {
                return Err(invalid_assistant(
                    "Muse invocation has a different capture contract",
                ))
            }
        };
        let placement = super::capture_placement::PreparedTargetCapturePlacement::prepare(
            execution,
            &descriptor,
            "cold-target-capture-admission",
            paths,
            |maximum_ranks| {
                execution.component_partition_layouts(&descriptor, &parameters, maximum_ranks)
            },
        )
        .map_err(invalid_assistant)?;
        // Prefill can consume the whole selected decoder invocation; replay and
        // shorter verification calls remain covered by the same envelope.
        placement
            .resolve([1, maximum_sequence as u64])
            .map_err(invalid_assistant)?;
        NonZeroUsize::new(maximum_sequence as usize)
            .map(AdmittedTargetInvocations::Coordinated)
            .ok_or_else(|| invalid_assistant("target invocation sequence must be positive"))
    }

    /// Selects source geometry and checks retained target reduction bounds before
    /// opening SafeTensors or consuming a retained GGUF source. This does not
    /// admit a speculative session.
    pub fn select_target_placement(
        mut self,
        target: &crate::SelectedPreparation,
    ) -> Result<Self, ArtifactError> {
        let rank = target_rank(target, &self.target, self.placement)?;
        let mut placement = ExternalAssistantSourcePlacement::Replicated(rank);
        match &mut self.preparation {
            DispatchedNormalizedExternalAssistant::Muse(construction) => {
                if let muse::assistant::execution::MuseAssistantPlacement::TensorParallel(rank) =
                    construction
                        .module_config
                        .execution_definition()
                        .target_placement(rank)
                {
                    let (source, selected, layout) = partition_muse_construction(
                        &construction.source_module_config,
                        &construction.module_config,
                        &construction.selected.tasks,
                        rank,
                    )?;
                    validate_tensor_group(target, &selected)?;
                    construction.source_module_config = source;
                    construction.module_config = selected;
                    construction.parameter_layout = Some(layout);
                    placement = ExternalAssistantSourcePlacement::TensorParallel(rank);
                }
            }
            DispatchedNormalizedExternalAssistant::Gemma4(_) if rank.world_size() != 1 => {
                return Err(invalid_assistant(
                    "distributed Gemma assistant source placement is not implemented",
                ));
            }
            DispatchedNormalizedExternalAssistant::Gemma4(_) => {}
        }
        self.placement = Some(placement);
        self.target_selection = Some(AssistantTargetSelection::Placed(target.clone()));
        Ok(self)
    }

    /// Source policy retained before the prepared source handoff.
    pub const fn source_placement(&self) -> Option<ExternalAssistantSourcePlacement> {
        self.placement
    }
}
