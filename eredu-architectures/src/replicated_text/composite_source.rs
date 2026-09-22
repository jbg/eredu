//! Existing composite declarations retained by their exact source/selection owner.
use super::*;
use crate::prepared_sources::PreparedModelSources;
use eredu_nn::workspace::WorkspaceMetadataError;

pub(super) enum Requirements {
    Owned(SharedCompositeConfig<CompositeTextRequirements>),
    Prepared(PreparedModelSources),
}
impl std::ops::Deref for Requirements {
    type Target = CompositeTextRequirements;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(value) => value,
            Self::Prepared(source) => source
                .composite_requirements()
                .expect("immutable prepared composite requirements were authenticated"),
        }
    }
}
impl Requirements {
    // Ordinary entry points retain their actual moved requirements in the same
    // paid immutable owner as model configurations. The prepared variant remains
    // its existing source alias and no longer carries this inactive payload.
    pub(super) fn owned<B: NeuralBackend>(
        value: CompositeTextRequirements,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, eredu_nn::Error> {
        let metadata = B::construction_metadata(context);
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                Self,
                Result<Self, eredu_nn::Error>,
                Option<&eredu_nn::workspace::WorkspaceContext>,
            )>())?;
        }
        SharedCompositeConfig::new(value, metadata).map(Self::Owned)
    }

    pub(super) fn prepared<B: NeuralBackend, E>(
        source: &PreparedModelSources,
        selected: &SelectedCompositeTextRealization,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, ReplicatedTextDispatchError<E>> {
        if source.composite_requirements().is_none()
            || !config_source::exact_selection(source, selected.execution())
        {
            return Err(ReplicatedTextDispatchError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        if let Some(context) = B::construction_metadata(context) {
            let controls = [
                size_of::<Self>(),
                size_of::<PreparedModelSources>(),
                size_of::<Result<Self, ReplicatedTextDispatchError<E>>>(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or_else(|| {
                    ReplicatedTextDispatchError::Metadata(WorkspaceMetadataError::Overflow.into())
                })?;
            context
                .charge_metadata(bytes)
                .map_err(|cause| ReplicatedTextDispatchError::Metadata(cause.into()))?;
        }
        Ok(Self::Prepared(source.clone()))
    }

    pub(super) fn source(&self) -> Option<&PreparedModelSources> {
        match self {
            Self::Owned(_) => None,
            Self::Prepared(source) => Some(source),
        }
    }
}

// Only actual configuration owners from completed model construction enter this
// slot. They contain no source alias, so the source cannot retain itself.
#[derive(Clone)]
pub(super) enum Configurations {
    Gemma4 {
        target: crate::gemma4::model::RetainedModelSource,
        source: Option<crate::gemma4::model::RetainedModelSource>,
    },
    Inkling {
        target: crate::inkling::RetainedModelSource,
        source: Option<crate::inkling::RetainedModelSource>,
    },
    QwenVl {
        target: SharedCompositeConfig<crate::qwen::vl::ModelArgs>,
        source: Option<SharedCompositeConfig<crate::qwen::vl::ModelArgs>>,
    },
    QwenHybrid {
        target: SharedCompositeConfig<crate::qwen::hybrid::ParsedHybridConfig>,
        source: Option<SharedCompositeConfig<crate::qwen::hybrid::ParsedHybridConfig>>,
        target_units: crate::qwen::hybrid::RetainedConditionalUnits,
        source_units: Option<crate::qwen::hybrid::RetainedConditionalUnits>,
    },
}
#[derive(Clone)]
pub(super) struct Completed {
    pub(super) configurations: Configurations,
    pub(super) capability: crate::capability::CapabilityEstimate,
}

impl Requirements {
    pub(super) fn completed<B: NeuralBackend, E>(
        &self,
        selected: &SelectedCompositeTextRealization,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Completed>, ReplicatedTextDispatchError<E>> {
        let Some(source) = self.source() else {
            return Ok(None);
        };
        if !config_source::exact_selection(source, selected.execution()) {
            return Err(ReplicatedTextDispatchError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        let metadata = B::construction_metadata(context);
        if let Some(metadata) = metadata {
            let controls = [
                size_of::<Completed>(),
                size_of::<Option<Completed>>(),
                size_of::<Option<Configurations>>(),
                size_of::<Result<Option<Completed>, ReplicatedTextDispatchError<E>>>(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or_else(|| {
                    ReplicatedTextDispatchError::Metadata(WorkspaceMetadataError::Overflow.into())
                })?;
            metadata
                .charge_metadata(bytes)
                .map_err(|cause| ReplicatedTextDispatchError::Metadata(cause.into()))?;
        }
        let completed = source.construction_semantics().composite.get();
        if completed.is_none() && metadata.is_some_and(|value| value.uses_checked_metadata()) {
            return Err(ReplicatedTextDispatchError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        Ok(completed.cloned())
    }

    pub(super) fn publish<B: NeuralBackend, E>(
        &self,
        selected: &SelectedReplicatedTextRealization,
        configurations: Option<Configurations>,
        capability: &crate::capability::CapabilityEstimate,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextDispatchError<E>> {
        // A workspace reconstruction never becomes a source of native initial
        // configuration ownership. Publication follows successful full contract
        // construction in the ordinary native handoff only.
        if B::construction_metadata(context).is_some() {
            return Ok(());
        }
        let (Some(source), Some(configurations)) = (self.source(), configurations) else {
            return Ok(());
        };
        if !config_source::exact_selection(source, selected) {
            return Err(ReplicatedTextDispatchError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        let completed = source
            .construction_semantics()
            .composite
            .get_or_init(|| Completed {
                configurations,
                capability: capability.clone(),
            });
        if &completed.capability != capability {
            return Err(ReplicatedTextDispatchError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        Ok(())
    }
}

/// Move the public owned configuration, or consume an existing closed alias,
/// through the same model constructor. No mutable or raw shared handle escapes.
pub(crate) enum ModelConfig<C> {
    Owned(C),
    Retained(SharedCompositeConfig<C>),
}
impl<C> std::ops::Deref for ModelConfig<C> {
    type Target = C;
    fn deref(&self) -> &C {
        match self {
            Self::Owned(value) => value,
            Self::Retained(value) => value,
        }
    }
}
impl<C> ModelConfig<C> {
    pub(crate) fn into_shared(
        self,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<SharedCompositeConfig<C>, eredu_nn::Error> {
        match self {
            Self::Owned(value) => SharedCompositeConfig::new(value, metadata),
            Self::Retained(value) => Ok(value),
        }
    }
}

impl Requirements {
    /// Only the actual completed typed unit source can open its constructor.
    /// Config equality or a bank description alone is insufficient.
    pub(super) fn validate_completed_units(&self) -> Result<(), eredu_nn::Error> {
        let completed = self
            .source()
            .and_then(|source| source.construction_semantics().composite.get())
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        match &completed.configurations {
            Configurations::Gemma4 { .. } => Ok(()),
            Configurations::Inkling { target, source }
                if target.has_units()
                    && source.as_ref().is_none_or(|source| source.has_units()) =>
            {
                Ok(())
            }
            Configurations::QwenHybrid {
                source,
                source_units,
                ..
            } if source.is_some() == source_units.is_some() => Ok(()),
            _ => Err(WorkspaceMetadataError::Unqualified.into()),
        }
    }
}

impl PreparedModelSources {
    pub(crate) fn matches_retained_inkling_admission(
        &self,
        admission: &crate::inkling::ModelArgs,
    ) -> bool {
        self.construction_semantics().composite.get().is_some_and(|source| {
            matches!(&source.configurations, Configurations::Inkling { target, .. } if target.matches_admission(admission))
        }) || self.construction_semantics().direct_partition.get().is_some_and(|source| source.matches_inkling_admission(admission))
    }
    pub(crate) fn matches_retained_gemma_admission(
        &self,
        admission: &crate::gemma4::FamilyConfig,
    ) -> bool {
        self.construction_semantics().composite.get().is_some_and(|source| {
            matches!(&source.configurations, Configurations::Gemma4 { target, .. } if target.matches_admission(admission))
        }) || self.construction_semantics().direct_partition.get().is_some_and(|source| source.matches_gemma_admission(admission))
    }
}
