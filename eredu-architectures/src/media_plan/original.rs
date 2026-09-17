//! Original semantic compilation over actual prepared model and host sources.
mod config;
mod encoder;
mod gemma;
mod position_facts;
use super::qwen::{
    self, InspectedPartRef, MediaSemanticError, QwenPartRef, QwenPartRole, QwenPolicy,
};
use crate::{
    prepared_execution::PreparedInferenceBlueprint,
    prepared_sources::PreparedModelSources,
    qwen::vl::positions::{
        GridRows, PositionComponent, PositionDestination, emit_positions, validate_positions,
    },
};
pub use encoder::PreparedMediaEncoderTablePlan;
use eredu_core::{InputMetadataKey, InputModality};
use eredu_runtime::{
    input::host::{HostInputPartView, HostTensorValues},
    working_memory::{
        BoundCompositeSemanticStorage, CompositeSemanticCoordinates, CompositeSemanticPartRecord,
        CompositeSemanticRole, MediaSessionBinding, OriginalCompositeSemanticStorage,
        OriginalCompositeSemanticStorageError, OriginalPreparedHostInput,
        PreparedCompositeSemanticRecipe, WorkingMemoryError, WorkingMemoryPool,
    },
};
pub use position_facts::{OriginalPromptSegment, PreparedMediaPositionFacts};

#[derive(Clone, Copy)]
enum Policy<'a> {
    Vl(&'a crate::qwen::vl::ModelArgs),
    Gemma(&'a crate::gemma4::FamilyConfig),
    Conditional(&'a crate::qwen::hybrid::ParsedHybridConfig),
}
impl<'a> Policy<'a> {
    fn source(sources: &'a PreparedModelSources) -> Result<Self, MediaSemanticError> {
        use crate::{
            configuration::{GgufModelConfig as G, SafetensorsModelConfig as S},
            gguf_companion::GgufMediaProjectorConfig as P,
        };
        let architecture = sources.architecture();
        if let Some(projector) = architecture.gguf_media_projector() {
            return match projector.model() {
                P::Qwen3Vl(args) => Ok(Self::Vl(args)),
                P::Gemma4(args) => Ok(Self::Gemma(args)),
                P::Qwen35(args) => Ok(Self::Conditional(args)),
                _ => Err(MediaSemanticError::input(
                    "source has no resolved original media semantic policy",
                )),
            };
        }
        match (
            architecture.safetensors_architecture().map(|p| p.model()),
            architecture.gguf_plan().map(|p| p.model()),
        ) {
            (Some(S::QwenVl(args)), None) => Ok(Self::Vl(args)),
            (Some(S::Gemma4(args)), None) | (None, Some(G::Gemma4(args))) => Ok(Self::Gemma(args)),
            (Some(S::QwenHybrid(args)), None) | (None, Some(G::QwenHybrid(args)))
                if args.vision.is_some() =>
            {
                Ok(Self::Conditional(args))
            }
            _ => Err(MediaSemanticError::input(
                "source has no original media semantic policy",
            )),
        }
    }
    fn qwen(self) -> QwenPolicy<'a> {
        match self {
            Self::Gemma(_) => unreachable!("typed Gemma source does not consume Qwen encoder tables"),
            Self::Vl(args) => QwenPolicy {
                hidden: args.text.hidden_size,
                vision: Some(&args.vision),
                image: Some(args.image_token_id),
                video: Some(args.video_token_id),
                projected_media: false,
            },
            Self::Conditional(args) => QwenPolicy {
                hidden: args.text.hidden_size,
                vision: args.vision.as_ref(),
                image: args.image_token_id,
                video: args.video_token_id,
                projected_media: true,
            },
        }
    }
    fn coordinates(self) -> CompositeSemanticCoordinates {
        match self {
            Self::Vl(_) => CompositeSemanticCoordinates::ThreeAxesAndPrefix,
            Self::Conditional(_) | Self::Gemma(_) => CompositeSemanticCoordinates::Ordinary,
        }
    }
}
fn parts<'a>(
    source: &'a OriginalPreparedHostInput,
    policy: QwenPolicy<'a>,
) -> impl Iterator<Item = Result<QwenPartRef<'a>, MediaSemanticError>> + Clone {
    source.parts().enumerate().map(move |(index, part)| {
        qwen::qwen_part(policy, InspectedPartRef::original(part)?).map_err(|error| error.at(index))
    })
}
struct SemanticPart<'a> {
    role: CompositeSemanticRole,
    modality: InputModality,
    positions: u64,
    placeholder: u32,
    grid: &'a [[i32; 3]],
    workspace_scalars: u64,
}
fn semantic_part<'a>(
    part: eredu_runtime::input::host::PreparedHostPart<'a>,
    policy: Policy<'a>,
) -> Result<SemanticPart<'a>, MediaSemanticError> {
    if let Policy::Gemma(args) = policy {
        let modality = part.modality();
        let plan = super::admission::gemma::raw::original_part(args, &part)?;
        let (role, positions, placeholder, workspace_scalars) = match plan {
            super::Gemma4InputPartPlan::TextTokens { positions } => (CompositeSemanticRole::Tokens, positions, 0, 0),
            super::Gemma4InputPartPlan::Projected { positions, placeholder_token_id, .. } => (CompositeSemanticRole::Projected, positions, placeholder_token_id, 0),
            super::Gemma4InputPartPlan::Vision { placeholder_token_id, shape, .. }
            | super::Gemma4InputPartPlan::Audio { placeholder_token_id, shape, .. } =>
                (CompositeSemanticRole::Encoded, shape.decoder_positions, placeholder_token_id, shape.execution_workspace_scalars),
        };
        let grid = if role == CompositeSemanticRole::Encoded && matches!(modality, InputModality::Image | InputModality::Video) {
            let (_, view) = part.metadata_view(InputMetadataKey::PatchGrid).ok_or(MediaSemanticError::input("Gemma original source grid"))?;
            let HostTensorValues::I32(values) = view.values else { return Err(MediaSemanticError::input("Gemma original source grid type")); };
            let (rows, tail) = values.as_chunks::<3>();
            if !tail.is_empty() { return Err(MediaSemanticError::input("Gemma original source grid rows")); }
            rows
        } else { &[] };
        return Ok(SemanticPart { role, modality, positions, placeholder, grid, workspace_scalars });
    }
    let value = qwen::qwen_part(policy.qwen(), InspectedPartRef::original(part)?)?;
    Ok(SemanticPart {
        role: match value.role { QwenPartRole::Tokens => CompositeSemanticRole::Tokens,
            QwenPartRole::Projected => CompositeSemanticRole::Projected, QwenPartRole::Encoded => CompositeSemanticRole::Encoded },
        modality: value.modality, positions: value.positions, placeholder: value.placeholder,
        grid: value.grid, workspace_scalars: value.workspace_scalars,
    })
}
fn semantic_parts<'a>(source: &'a OriginalPreparedHostInput, policy: Policy<'a>)
    -> impl Iterator<Item = Result<SemanticPart<'a>, MediaSemanticError>> + Clone {
    source.parts().enumerate().map(move |(index, part)| semantic_part(part, policy).map_err(|cause| cause.at(index)))
}
fn positions<'a>(
    source: &'a OriginalPreparedHostInput,
    policy: QwenPolicy<'a>,
) -> impl Iterator<Item = Result<PositionComponent<'a>, MediaSemanticError>> + Clone {
    parts(source, policy).map(|part| {
        part.and_then(|part| match part.role {
            QwenPartRole::Encoded => Ok(PositionComponent::Media(GridRows::Arrays(part.grid))),
            _ => i64::try_from(part.positions)
                .map(PositionComponent::Text)
                .map_err(|_| MediaSemanticError::overflow("Qwen decoder positions")),
        })
    })
}
fn account(error: WorkingMemoryError) -> MediaSemanticError {
    match error {
        WorkingMemoryError::Overflow => {
            MediaSemanticError::overflow("original media semantic layout")
        }
        _ => MediaSemanticError::input("original media semantic storage geometry"),
    }
}

/// ```compile_fail
/// use eredu_architectures::prepared_sources::PreparedModelSources;
/// use eredu_runtime::working_memory::{OriginalPreparedHostInput,WorkingMemoryPool};
/// fn no_detach(sources:PreparedModelSources,source:&OriginalPreparedHostInput,pool:&WorkingMemoryPool) {
///     let plan=sources.plan_original_media_semantics(source).unwrap();
///     drop(sources);
///     let _=plan.compile(pool);
/// }
/// ```
/// ```compile_fail
/// use eredu_architectures::media_plan::PreparedMediaSemanticCompile;
/// use eredu_runtime::working_memory::WorkingMemoryPool;
/// fn once(plan:PreparedMediaSemanticCompile<'_, '_>,pool:&WorkingMemoryPool) {
///     let _=plan.compile(pool);
///     let _=plan.compile(pool);
/// }
/// ```
/// Consuming checked recipe. It borrows the actual selected sources through
/// compilation and binding; no source graph/configuration clone is made here.
#[repr(transparent)]
pub struct PreparedMediaSemanticCompile<'s, 'h>(
    PreparedCompositeSemanticRecipe<'s, 'h, PreparedModelSources>,
);
/// Closed original semantic body plus the still-live cold source-selection loan.
/// ```compile_fail
/// use eredu_architectures::media_plan::OriginalPreparedMediaSemantics;
/// fn no_mutation(mut source:OriginalPreparedMediaSemantics<'_>) {
///     source.coordinates_mut()[0]=3;
/// }
/// ```
/// It cannot be relabelled, modified, deserialized, or detached from that loan.
#[repr(transparent)]
pub struct OriginalPreparedMediaSemantics<'s>(
    OriginalCompositeSemanticStorage<'s, PreparedModelSources>,
);
/// Compiled semantics authenticated against a current actual native destination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticKind { Vl, Conditional, Gemma }
impl Policy<'_> {
    fn kind(self) -> SemanticKind { match self {
        Self::Vl(_) => SemanticKind::Vl,
        Self::Conditional(_) => SemanticKind::Conditional,
        Self::Gemma(_) => SemanticKind::Gemma,
    } }
}
#[derive(Clone)]
pub struct BoundPreparedMediaSemantics(BoundCompositeSemanticStorage, SemanticKind);
impl std::fmt::Debug for BoundPreparedMediaSemantics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BoundPreparedMediaSemantics { .. }")
    }
}
impl PreparedModelSources {
    pub fn plan_original_media_semantics<'s, 'h>(
        &'s self,
        source: &'h OriginalPreparedHostInput,
    ) -> Result<PreparedMediaSemanticCompile<'s, 'h>, MediaSemanticError> {
        let policy = Policy::source(self)?;
        let processor =
            self.selected()
                .execution()
                .processor()
                .ok_or(MediaSemanticError::input(
                    "selected execution has no composite processor",
                ))?;
        let mut total = 0u64;
        let mut encoded = false;
        for (index, part) in source.parts().enumerate() {
            validate_selected_part(processor, part.modality(), part.kind())
                .map_err(|error| error.at(index))?;
            let part = semantic_part(part, policy).map_err(|error| error.at(index))?;
            total = total
                .checked_add(part.positions)
                .ok_or(MediaSemanticError::overflow(
                    "Qwen complete decoder positions",
                ))?;
            encoded |= part.role == CompositeSemanticRole::Encoded;
        }
        if !encoded || total == 0 {
            return Err(MediaSemanticError::input(
                "retained media requires a positive source and actual raw encoder part",
            ));
        }
        let count = usize::try_from(total)
            .map_err(|_| MediaSemanticError::overflow("Qwen decoder positions"))?;
        if let Policy::Vl(args) = policy {
            validate_positions(
                positions(source, policy.qwen()),
                args.vision.spatial_merge_size,
                count,
            )?;
        }
        let recipe =
            PreparedCompositeSemanticRecipe::new(self, source, count, policy.coordinates())
                .map_err(account)?;
        recipe.required_bytes().map_err(account)?;
        Ok(PreparedMediaSemanticCompile(recipe))
    }
}
/// Shared selection check; the existing native adapter supplies only the actual
/// per-part tags. It does not reproduce processor-selection policy.
pub fn validate_selected_part(
    processor: &eredu_runtime::SelectedProcessorExecution,
    modality: InputModality,
    kind: eredu_core::InputPayloadKind,
) -> Result<(), MediaSemanticError> {
    if !processor.modalities().contains(&modality) {
        return Err(MediaSemanticError::input(
            "prepared modality is outside selected composite modalities",
        ));
    }
    if modality != InputModality::Text && !processor.prepared_tensors() {
        return Err(MediaSemanticError::input(
            "prepared media tensors were not admitted by processor selection",
        ));
    }
    if kind == eredu_core::InputPayloadKind::Embeddings
        && !processor.projected_modalities().contains(&modality)
    {
        return Err(MediaSemanticError::input(
            "projected modality was not admitted by processor selection",
        ));
    }
    Ok(())
}
impl<'s, 'h> PreparedMediaSemanticCompile<'s, 'h> {
    pub fn required_bytes(&self) -> Result<u64, MediaSemanticError> {
        self.0.required_bytes().map_err(account)
    }
    pub fn decoder_positions(&self) -> usize {
        self.0.layout().positions()
    }
    pub fn compile(
        self,
        pool: &WorkingMemoryPool,
    ) -> Result<OriginalPreparedMediaSemantics<'s>, OriginalCompositeSemanticStorageError> {
        let source = self.0.source();
        let provenance = self.0.provenance();
        let layout = self.0.layout();
        let policy = Policy::source(provenance).expect("immutable preflighted policy");
        let mut builder = self.0.allocate(pool)?;
        let mut position = 0u64;
        for (index, result) in semantic_parts(source, policy).enumerate() {
            let part = match result {
                Ok(part) => part,
                Err(error) => return Err(builder.fail_semantic(error.diagnostic())),
            };
            let end = match position.checked_add(part.positions) {
                Some(end) => end,
                None => {
                    return Err(builder.fail_semantic(
                        MediaSemanticError::overflow("Qwen complete decoder positions")
                            .diagnostic(),
                    ));
                }
            };
            let grid_slot = source
                .part(index)
                .and_then(|part| part.metadata_view(InputMetadataKey::PatchGrid))
                .map(|(slot, _)| slot);
            builder.records_mut()[index] = CompositeSemanticPartRecord {
                source_part: index,
                grid_slot,
                grid_rows: part.grid.len(),
                start: position,
                end,
                role: part.role,
                modality: part.modality,
                placeholder: part.placeholder,
                workspace_scalars: part.workspace_scalars,
            };
            position = end;
        }
        if let Policy::Vl(args) = policy {
            let count = layout.positions();
            let coordinates = builder.coordinates_mut();
            let (first, rest) = coordinates.split_at_mut(count);
            let (second, rest) = rest.split_at_mut(count);
            let (third, prefix) = rest.split_at_mut(count);
            if let Err(error) = emit_positions(
                positions(source, policy.qwen()),
                args.vision.spatial_merge_size,
                count,
                PositionDestination {
                    axes: [first, second, third],
                    prefix: Some(prefix),
                },
            ) {
                return Err(builder.fail_semantic(error.diagnostic()));
            }
        }
        builder.finish().map(OriginalPreparedMediaSemantics)
    }
}
impl<'s> OriginalPreparedMediaSemantics<'s> {
    /// Consumes this source into a typed owning boundary failure. This issues no authority.
    pub fn reject_boundary(
        self,
        error: WorkingMemoryError,
    ) -> OriginalCompositeSemanticStorageError {
        self.0.reject_boundary(error)
    }

    pub(crate) fn reject(self, error: MediaSemanticError) -> OriginalCompositeSemanticStorageError {
        self.0.reject(error.diagnostic())
    }
    pub fn source(&self) -> &OriginalPreparedHostInput {
        self.0.source()
    }
    pub fn decoder_positions(&self) -> usize {
        self.0.layout().positions()
    }
    pub fn original_bytes(&self) -> u64 {
        self.0.original_bytes()
    }
    pub fn matches_destination(&self, blueprint: &PreparedInferenceBlueprint) -> bool {
        blueprint.has_selected_sources(self.0.provenance())
    }
    pub(crate) fn bind_vl(
        self,
        admission: &crate::qwen::vl::ModelArgs,
        blueprint: &PreparedInferenceBlueprint,
        source: &OriginalPreparedHostInput,
        binding: MediaSessionBinding,
    ) -> Result<BoundPreparedMediaSemantics, OriginalCompositeSemanticStorageError> {
        let matches = matches!(Policy::source(self.0.provenance()),Ok(Policy::Vl(args)) if config::vl(args,admission));
        self.bind_checked(matches, blueprint, source, binding)
    }
    pub(crate) fn bind_conditional(
        self,
        admission: &crate::qwen::hybrid::ParsedHybridConfig,
        blueprint: &PreparedInferenceBlueprint,
        source: &OriginalPreparedHostInput,
        binding: MediaSessionBinding,
    ) -> Result<BoundPreparedMediaSemantics, OriginalCompositeSemanticStorageError> {
        let matches = matches!(Policy::source(self.0.provenance()),Ok(Policy::Conditional(args)) if config::conditional(args,admission));
        self.bind_checked(matches, blueprint, source, binding)
    }
    pub(crate) fn bind_gemma(
        self,
        admission: &crate::gemma4::FamilyConfig,
        blueprint: &PreparedInferenceBlueprint,
        source: &OriginalPreparedHostInput,
        binding: MediaSessionBinding,
    ) -> Result<BoundPreparedMediaSemantics, OriginalCompositeSemanticStorageError> {
        // The completed constructor retained this exact immutable admission
        // owner after source-format normalization. Equal configs are insufficient.
        let matches = matches!(Policy::source(self.0.provenance()), Ok(Policy::Gemma(_)))
            && self.0.provenance().matches_retained_gemma_admission(admission);
        self.bind_checked(matches, blueprint, source, binding)
    }
    fn bind_checked(
        self,
        config_matches: bool,
        blueprint: &PreparedInferenceBlueprint,
        source: &OriginalPreparedHostInput,
        binding: MediaSessionBinding,
    ) -> Result<BoundPreparedMediaSemantics, OriginalCompositeSemanticStorageError> {
        if !config_matches
            || !self.matches_destination(blueprint)
            || !self.source().same_source(source)
            || binding.frontier() != 0
        {
            return Err(self.0.reject(
                MediaSemanticError::input(
                    "original media source/selection/current-state binding mismatch",
                )
                .diagnostic(),
            ));
        }
        let kind = Policy::source(self.0.provenance()).expect("validated semantic policy").kind();
        Ok(BoundPreparedMediaSemantics(self.0.bind(binding), kind))
    }
}
impl BoundPreparedMediaSemantics {
    /// Carries the same admitted source/configuration through the actual shared
    /// copied-state exchange. The receipt cannot be minted from a raw frontier.
    pub fn for_copied_state(
        &self,
        transition: &eredu_runtime::working_memory::CopiedMediaStateBinding,
    ) -> Result<Self, eredu_runtime::working_memory::WorkingMemoryError> {
        self.0.for_copied_state(transition).map(|storage| Self(storage, self.1))
    }

    /// Borrows original ordered source attribution without token/grid copies,
    /// semantic re-admission, native evaluation or a new binding attempt.
    pub fn request_position_facts(
        &self,
    ) -> Result<PreparedMediaPositionFacts<'_>, MediaSemanticError> {
        PreparedMediaPositionFacts::from_source(
            self.source(),
            self.records(),
            self.decoder_positions(),
        )
    }

    pub(crate) fn reject(self, error: MediaSemanticError) -> OriginalCompositeSemanticStorageError {
        self.0.reject(error.diagnostic())
    }
    pub fn source(&self) -> &OriginalPreparedHostInput {
        self.0.source()
    }
    pub fn decoder_positions(&self) -> usize {
        self.0.layout().positions()
    }
    pub fn original_bytes(&self) -> u64 {
        self.0.original_bytes()
    }
    pub fn binding(&self) -> &MediaSessionBinding {
        self.0.binding()
    }
    pub(crate) fn records(&self) -> &[CompositeSemanticPartRecord] {
        self.0.records()
    }
    pub(crate) fn coordinates(&self) -> &[i32] {
        self.0.coordinates()
    }
    pub(crate) fn part(&self, index: usize) -> QwenPartRef<'_> {
        assert!(self.1 != SemanticKind::Gemma, "typed Qwen admission source");
        let record = &self.0.records()[index];
        let grid = if record.role == CompositeSemanticRole::Encoded {
            let part = self
                .source()
                .part(record.source_part)
                .expect("compiled source part");
            let (slot, view) = part
                .metadata_view(InputMetadataKey::PatchGrid)
                .expect("compiled source grid");
            assert_eq!(Some(slot), record.grid_slot);
            let HostTensorValues::I32(values) = view.values else {
                unreachable!("compiled I32 grid")
            };
            let (rows, tail) = values.as_chunks::<3>();
            assert!(tail.is_empty());
            assert_eq!(rows.len(), record.grid_rows);
            rows
        } else {
            &[]
        };
        QwenPartRef {
            role: match record.role {
                CompositeSemanticRole::Tokens => QwenPartRole::Tokens,
                CompositeSemanticRole::Projected => QwenPartRole::Projected,
                CompositeSemanticRole::Encoded => QwenPartRole::Encoded,
            },
            modality: record.modality,
            positions: record.end - record.start,
            placeholder: record.placeholder,
            grid,
            workspace_scalars: record.workspace_scalars,
        }
    }
}
