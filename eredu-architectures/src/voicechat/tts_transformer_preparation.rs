//! Cold exact binding plans for independently leased TTS transformer units.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{RecipeDtype, RecipeError},
    store::{SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    ComponentPreparationError, LocalModelLayout, PreparedComponent, RecipeBindingError,
    WeightBinding,
};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
    parameters::{admitted_member, execution_parameter, SpeechParameterError},
    tts_transformer::TransformerConfig,
    tts_transformer_layered::TransformerError,
    tts_transformer_specs::{released_declarations, TransformerDeclaration},
};

/// Selects one independently executed EAR-TTS transformer from retained artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtsTransformerKind {
    /// Causal speech backbone.
    Causal,
    /// Bidirectional character encoder.
    CharacterEncoder,
}

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum TtsTransformerPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Invalid floating execution profile or attempted precision reselection.
    #[error("invalid speech precision: {0}")]
    Precision(&'static str),
    /// Transformer geometry cannot be constructed.
    #[error(transparent)]
    Transformer(#[from] TransformerError),
    /// A required parameter is missing from the selected component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// Source restriction failed.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// A recipe disagrees with its exact source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Exact recipe binding cannot be represented.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat TTS transformer parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for each decoder block and final normalization.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedTtsTransformerBindings {
    prepared: PreparedComponent<TransformerDeclaration>,
}

impl PreparedTtsTransformerBindings {
    /// Selects exact narrow projections; all norms and cache storage remain F32.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, TtsTransformerPreparationError> {
        if self.config().precision() != Default::default() && self.config().precision() != precision
        {
            return Err(TtsTransformerPreparationError::Precision(
                "precision is already selected",
            ));
        }
        precision
            .validate()
            .map_err(TtsTransformerPreparationError::Precision)?;
        let declaration = super::precision::SpeechPrecisionDeclaration::from_groups(
            self.definition().parameters().groups(),
        );
        self.prepared = self.prepared.transform_units(|source, units| {
            precision
                .bindings(source, units, &declaration)
                .map_err(TtsTransformerPreparationError::from)
        })?;
        let declaration = self.declaration().clone().with_precision(precision)?;
        self.prepared = self
            .prepared
            .project("voicechat/tts_transformer/leased-units", declaration)?;
        Ok(self)
    }

    /// Projects exact transformer bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
        kind: TtsTransformerKind,
    ) -> Result<Self, TtsTransformerPreparationError> {
        let source = artifacts
            .component_source(Component::Tts)
            .ok_or(SpeechParameterError::MissingComponent(Component::Tts))?;
        Self::from_parameters(
            match kind {
                TtsTransformerKind::Causal => released_declarations()?.causal.clone(),
                TtsTransformerKind::CharacterEncoder => released_declarations()?.character.clone(),
            },
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring TTS component payloads are excluded.
    pub fn from_parameters(
        declaration: TransformerDeclaration,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, TtsTransformerPreparationError> {
        use eredu_nn::DeclareLogicalParameters;
        let formats = declaration
            .spec()
            .logical_parameters()
            .map_err(TransformerError::from)?
            .into_iter()
            .filter_map(|slot| {
                let name = slot.parameter.id.as_str();
                parameters
                    .get(name)
                    .filter(|p| p.component == Component::Tts)
                    .map(|p| (name.to_owned(), p.linear_format))
            })
            .collect();
        let declaration = declaration.with_formats(formats)?;
        let prepared = PreparedComponent::prepare(
            "voicechat/tts_transformer/leased-units",
            declaration,
            source.clone(),
            |member| {
                let name = member.target();
                let parameter = admitted_member(parameters, &source, Component::Tts, member)?;
                let parameter = execution_parameter(parameter, &source)?;
                if !matches!(parameter.output.dtype, RecipeDtype::F32 | RecipeDtype::U32)
                    && !(parameter.output.dtype == RecipeDtype::U8
                        && super::parameters::is_mxfp4_scale(parameters, member))
                {
                    return Err(TtsTransformerPreparationError::Geometry(name.into()));
                }
                Ok::<_, TtsTransformerPreparationError>(parameter.recipe)
            },
        )?;
        Ok(Self { prepared })
    }

    /// Selects complete-stack tensor placement using the shared physical planner.
    /// Component construction owns no pipeline transport or expert/data replicas;
    /// those dimensions require an enclosing architecture partition selection.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), TtsTransformerPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Applies an already selected rank-local layout to the exact retained recipes.
    /// Shape validation precedes recipe lowering and source authority is preserved.
    /// Reusing the same retained selection preserves its transformed recipes.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, TtsTransformerPreparationError> {
        let blocks =
            super::tts_transformer_layered::transformer_local_configs(self.declaration(), layout)?;
        let parameters = super::tts_transformer_layered::transformer_local_parameters(
            self.declaration(),
            &blocks,
        )?;
        let state =
            super::tts_transformer_layered::transformer_state_layout(self.config(), &parameters)?;
        let declaration = self.declaration().clone().with_state_layout(state)?;
        self.prepared = self
            .prepared
            .select(layout)?
            .project("voicechat/tts_transformer/leased-units", declaration)?;
        Ok(self)
    }

    /// Narrows retained source authority and exact bindings to one selected stage.
    /// Unit addresses remain canonical; this result is not a zero-based complete
    /// provider layout and must be consumed by a partition traversal.
    pub fn into_partition<G>(
        self,
        partition: &eredu_runtime::ArchitecturePartition<
            G,
            super::tts_transformer_layered::TransformerBoundarySchema,
        >,
    ) -> Result<PreparedTtsTransformerPartitionBindings, TtsTransformerPreparationError> {
        super::tts_transformer_layered::validate_transformer_partition(
            self.declaration(),
            self.prepared.local_layout(),
            partition,
        )?;
        let prepared = self.prepared.select_units(partition.units())?;
        Ok(PreparedTtsTransformerPartitionBindings {
            prepared,
            state_layout: partition.state().map(|state| state.layout().clone()),
        })
    }

    /// Typed scope retained through format, precision and state selection.
    pub fn declaration(&self) -> &TransformerDeclaration {
        self.prepared.declaration()
    }

    /// Exact retained tensor selection, when present.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated transformer geometry; no other speech stack is selected.
    pub fn config(&self) -> &TransformerConfig {
        self.prepared.declaration().config()
    }

    /// Exact prepared unit bindings for cold resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }

    /// Semantic definition with the selected rank's complete cache geometry.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration().definition()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        TransformerDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (declaration, selected, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("component declares only independently leased units");
        (declaration, selected, layout, source, units)
    }
}

/// Exact canonical units and restricted retained sources for a pipeline stage.
/// A normalization-only stage has no state and owns only its normalization recipe.
pub struct PreparedTtsTransformerPartitionBindings {
    prepared: PreparedComponent<TransformerDeclaration>,
    state_layout: Option<eredu_runtime::StateLayout>,
}
impl PreparedTtsTransformerPartitionBindings {
    /// Global scope retained alongside selected stage resources.
    pub fn declaration(&self) -> &TransformerDeclaration {
        self.prepared.declaration()
    }

    /// Global family equations; physical TP geometry is retained separately.
    pub fn config(&self) -> &TransformerConfig {
        self.prepared.declaration().config()
    }
    /// Exact selected physical layout when tensor parallelism is also selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }
    /// Source authority limited to the actual selected unit recipes.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }
    /// Canonical addresses and their exact resident/bounded materialization recipes.
    pub fn unit_bindings(
        &self,
    ) -> impl ExactSizeIterator<Item = (eredu_runtime::ExecutionUnitAddress, &[WeightBinding])>
    {
        self.prepared.addressed_unit_bindings()
    }
    /// Selected block cache schedule; absent for a standalone normalization stage.
    pub fn state_layout(&self) -> Option<&eredu_runtime::StateLayout> {
        self.state_layout.as_ref()
    }
    /// Complete definition with selected cache geometry and global unit identities.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration().definition()
    }
    /// Consumes exact stage authority without rebasing canonical unit addresses.
    pub fn into_parts(
        self,
    ) -> (
        TransformerDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
        Option<eredu_runtime::StateLayout>,
    ) {
        let (declaration, local_layout, invocations, source, units) = self
            .prepared
            .into_addressed_unit_parts()
            .expect("partition declares only independently leased units");
        (
            declaration,
            local_layout,
            invocations,
            source,
            units,
            self.state_layout,
        )
    }
}
