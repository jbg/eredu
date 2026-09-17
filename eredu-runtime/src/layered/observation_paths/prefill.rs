//! Architecture declarations retained by the same physical path owner.
use super::*;

/// Physical readout needed to make an observation's request rows available.
/// This classifies the hook's equation, not its path spelling or tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillReadoutStage {
    /// Input/body hooks run before hidden-position selection, including StateOnly.
    BeforeReadout,
    /// Actual selected readout input/residual/normalization. The present coupled
    /// worker requires Sequence; this does not grant a head-free implementation.
    ReadoutInput,
    /// Actual affine or final vocabulary scores for the selected request rows.
    VocabularyScores,
}

/// One architecture-issued ordinary-text hook declaration. Construction alone
/// supplies no attribution, native mechanism, budget or execution permission.
/// The actual runtime collects declarations under its private path binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillObservationDeclaration {
    path: String,
    sequence_axis: usize,
    stage: PrefillReadoutStage,
    flattened_tokens: bool,
}
impl PrefillObservationDeclaration {
    /// Declares causal request-row equivalence for the actual hook equation:
    /// splitting canonical ordinary text preserves each row under the same
    /// prefix/state/parameters (subject to normal selected numerical tolerance).
    /// No external masks, media/proposal invocation or activation intervention
    /// semantics are covered. A caller must not infer this from a Sequence axis.
    ///
    /// Generated projection input keeps its actual selected representation and
    /// factory; this declaration never replaces it by normalized hidden values.
    pub fn causal_ordinary_text(
        path: String,
        sequence_axis: usize,
        stage: PrefillReadoutStage,
    ) -> Self {
        Self {
            path,
            sequence_axis,
            stage,
            flattened_tokens: false,
        }
    }
    /// Declares decoder request-row equivalence under the actual validated
    /// retained-media ingress plan. Encoder/compact-feature axes and arbitrary
    /// masks or interventions are not covered. Stored separately from text.
    pub fn prepared_media_decoder(
        path: String,
        sequence_axis: usize,
        stage: PrefillReadoutStage,
    ) -> Self {
        Self {
            path,
            sequence_axis,
            stage,
            flattened_tokens: false,
        }
    }
    /// Declares the actual sparse expert equation to be causal and row-local:
    /// each provider token flattens the current [batch, new-token] input in
    /// batch-major order. Routing identities and scalar placement remain in the
    /// provider's source; this does not infer them from an axis or path suffix.
    /// No intervention, proposal invocation or media encoder semantics are granted.
    pub fn causal_routed_units(path: String) -> Self {
        Self {
            path,
            sequence_axis: 0,
            stage: PrefillReadoutStage::BeforeReadout,
            flattened_tokens: true,
        }
    }
    /// Whether the declared row coordinate flattens batch and new-token axes.
    pub const fn flattens_batch_tokens(&self) -> bool {
        self.flattened_tokens
    }
    /// Exact declared observation identity.
    pub fn path(&self) -> &str {
        &self.path
    }
    /// Source row axis; sparse declarations use flattened batch/token coordinates.
    pub const fn sequence_axis(&self) -> usize {
        self.sequence_axis
    }
    /// Actual hook timing relative to physical readout.
    pub const fn readout_stage(&self) -> PrefillReadoutStage {
        self.stage
    }
    pub(super) fn path_capacity(&self) -> usize {
        self.path.capacity()
    }
}
impl SharedLayeredObservationPaths {
    /// Exact declaration collected from the actual architecture. Ambiguous or
    /// absent declarations remain unknown; no first-match capability is inferred.
    /// This does not validate the runtime token or admit a selected capture.
    pub fn prefill_observation(&self, path: &str) -> Option<&PrefillObservationDeclaration> {
        let mut matches = self.0.prefill.iter().filter(|entry| entry.path() == path);
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }
}

/// Rejected semantic/source/candidate binding. These errors grant no fallback.
#[derive(Debug, thiserror::Error)]
pub enum PreparedCaptureSelectionError {
    /// Ordinary text identity/origin or actual source/path owner differs.
    #[error("capture row selection source or ordinary invocation differs")]
    Identity,
    /// No unique explicit causal declaration exists for this selected p0 hook.
    #[error("selected prefill hook {index} has no causal row declaration")]
    Undeclared {
        /// Exact selection ordinal.
        index: usize,
    },
    /// Declared architecture sequence axis differs from admitted shape semantics.
    #[error("selected prefill hook {index} has incompatible row axes")]
    Axes {
        /// Exact selection ordinal.
        index: usize,
    },
    /// Existing physical output omits rows required by the selected hook.
    #[error("capture row selection requires physical Sequence output")]
    Readout,
    /// Original admitted catalog/support no longer matches retained discovery.
    #[error(transparent)]
    Capture(#[from] eredu_core::capture::CaptureError),
    /// Candidate geometry is invalid before execution.
    #[error(transparent)]
    Geometry(#[from] eredu_core::CapabilityError),
    /// Closed logical-to-fragment mapping rejected the candidate.
    #[error(transparent)]
    Rows(#[from] eredu_core::capture::CapturePrefillGeometryError),
}

/// Retained immutable row/readout companion. Only existing physical source
/// aliases are owned: no selected table, copied discovery, native work or grant.
/// Architecture discovery must revalidate the admission before production use;
/// the installed session must also validate its existing runtime path token.
#[derive(Debug, Clone)]
pub struct PreparedCaptureSelection {
    source: eredu_core::capture::SharedCapturePlan,
    paths: SharedLayeredObservationPaths,
    sequence_readout: bool,
    media: bool,
}
impl SharedLayeredObservationPaths {
    /// Binds existing actual hook declarations to an exact admitted source.
    /// This checks row semantics only, not catalog support, original admission,
    /// current runtime token, parameter revision or native mechanism readiness.
    /// Architecture callers revalidate retained discovery before using it.
    pub fn prepare_capture_selection(
        &self,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<PreparedCaptureSelection, PreparedCaptureSelectionError> {
        if source.admission().text_origin().is_none() {
            return Err(PreparedCaptureSelectionError::Identity);
        }
        let mut selected = PreparedCaptureSelection {
            source: source.clone(),
            paths: self.clone(),
            sequence_readout: false,
            media: false,
        };
        for index in 0..source.admission().plan().selections.len() {
            if let Some(declaration) = selected.declaration(index)? {
                if declaration.readout_stage() != PrefillReadoutStage::BeforeReadout
                    && !matches!(
                        source.admission().plan().selections[index].transform,
                        eredu_core::capture::CaptureTransform::TopCandidates { .. }
                            | eredu_core::capture::CaptureTransform::TokenScores { .. }
                    )
                {
                    selected.sequence_readout = true;
                }
            }
        }
        Ok(selected)
    }
}
impl SharedLayeredObservationPaths {
    /// Ordinary source binding for decoder hooks of a validated media ingress.
    /// This allocates no payload and grants no original admission or execution.
    pub fn prepare_media_capture_selection(
        &self,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<PreparedCaptureSelection, PreparedCaptureSelectionError> {
        use eredu_core::capture::CaptureTransform;
        if source.admission().text_origin().is_none()
            || source.admission().invocation_bounds().is_some()
        {
            return Err(PreparedCaptureSelectionError::Identity);
        }
        let mut selected = PreparedCaptureSelection {
            source: source.clone(),
            paths: self.clone(),
            sequence_readout: false,
            media: true,
        };
        for (index, entry) in source.admission().plan().selections.iter().enumerate() {
            if !matches!(
                entry.transform,
                CaptureTransform::FullTensor
                    | CaptureTransform::Slice
                    | CaptureTransform::Summary
                    | CaptureTransform::Histogram { .. }
                    | CaptureTransform::Preview { .. }
                    | CaptureTransform::TopCandidates { .. }
                    | CaptureTransform::TokenScores { .. }
                    | CaptureTransform::RoutedUnits
            ) {
                return Err(PreparedCaptureSelectionError::Undeclared { index });
            }
            if let Some(declaration) = selected.declaration(index)? {
                selected.sequence_readout |= declaration.readout_stage()
                    != PrefillReadoutStage::BeforeReadout
                    && !matches!(
                        entry.transform,
                        CaptureTransform::TopCandidates { .. }
                            | CaptureTransform::TokenScores { .. }
                    );
            }
        }
        Ok(selected)
    }
    fn media_prefill_observation(&self, path: &str) -> Option<&PrefillObservationDeclaration> {
        let mut matches = self
            .0
            .media_prefill
            .iter()
            .filter(|entry| entry.path() == path);
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }
}

impl PreparedCaptureSelection {
    /// Semantic kind only; actual input/paths/revision must still be validated.
    pub fn is_prepared_media(&self) -> bool {
        self.media
    }
    /// Same immutable source owner used by original admission and host claims.
    pub fn source(&self) -> &eredu_core::capture::SharedCapturePlan {
        &self.source
    }
    /// Same loaded path/declaration owner; this is not its runtime token.
    pub fn paths(&self) -> &SharedLayeredObservationPaths {
        &self.paths
    }
    /// Conservative fixed construction/move overlap for this owner, its bound
    /// view, and the largest concrete row geometry built sequentially by
    /// `bind_geometry` (raw assembly, transform plan or candidate descriptor).
    /// Each geometry includes its actual fixed shape/stride storage.
    /// Both source payloads remain separately registered. Reconstructed declaration
    /// Vec/String buffers in cold path collection/rebinding remain separate original
    /// loading/preparation obligations; this fact does not bound those allocations.
    /// This is a control cost fact, not an allocation or execution grant.
    pub fn control_peak_bytes() -> Option<u64> {
        size_of::<Self>()
            .checked_add(size_of::<BoundCaptureSelection<'_>>())?
            .checked_add(
                size_of::<eredu_core::capture::CapturePrefillRowAssembly<'_>>()
                    .max(size_of::<
                        eredu_core::capture::CapturePrefillTransformPlan<'_>,
                    >())
                    .max(size_of::<eredu_core::capture::CaptureCandidateGeometry<'_>>())
                    .max(size_of::<eredu_core::capture::CaptureRoutedPrefillPlan<'_>>()),
            )?
            .checked_mul(3)?
            .try_into()
            .ok()
    }
    /// Read-only declaration for a scheduled p0 hook; None means inactive.
    /// Missing/ambiguous declarations reject, including zero Preview output.
    pub fn declaration(
        &self,
        index: usize,
    ) -> Result<Option<&PrefillObservationDeclaration>, PreparedCaptureSelectionError> {
        use eredu_core::{SymbolicDimension, capture::CapturePhase};
        let admission = self.source.admission();
        let selection = admission
            .plan()
            .selections
            .get(index)
            .ok_or(PreparedCaptureSelectionError::Identity)?;
        let point = admission
            .points()
            .get(index)
            .ok_or(PreparedCaptureSelectionError::Identity)?;
        if !point.prefill || !selection.schedule.includes(CapturePhase::Prefill, 0) {
            return Ok(None);
        }
        let declaration = if self.media {
            self.paths.media_prefill_observation(&point.path)
        } else {
            self.paths.prefill_observation(&point.path)
        }
        .ok_or(PreparedCaptureSelectionError::Undeclared { index })?;
        let axes = point
            .axes
            .as_ref()
            .ok_or(PreparedCaptureSelectionError::Axes { index })?;
        let sparse = matches!(point.value_type, eredu_core::ObservationValueType::RoutedUnits { .. });
        if sparse != declaration.flattens_batch_tokens()
            || (sparse && (axes.len() != 3
                || !matches!(selection.transform, eredu_core::capture::CaptureTransform::RoutedUnits)))
            || axes.len() > 32
            || axes.iter().enumerate().any(|(axis, value)| {
                if axis == declaration.sequence_axis() {
                    value.dimension != if sparse {
                        SymbolicDimension::TokenRows
                    } else {
                        SymbolicDimension::Sequence
                    }
                } else {
                    if sparse {
                        !matches!(value.dimension, SymbolicDimension::Known(_))
                    } else {
                        !matches!(value.dimension, SymbolicDimension::Batch | SymbolicDimension::Known(_))
                    }
                }
            })
            || declaration.sequence_axis() >= axes.len()
        {
            return Err(PreparedCaptureSelectionError::Axes { index });
        }
        Ok(Some(declaration))
    }
    /// Join physical readout before choosing/quoting a candidate. Sequence may
    /// still be required by other callers. This does not alter a bound geometry.
    pub fn physical_output(&self, minimum: eredu_core::OutputDemand) -> eredu_core::OutputDemand {
        if self.sequence_readout {
            eredu_core::OutputDemand::Sequence
        } else {
            minimum
        }
    }
    /// Bind exact candidate geometry without allocation. The original quote,
    /// request, source factory and canonical chunk stamps must all use this same
    /// physical output; this cannot retrofit an already admitted request.
    pub fn bind_geometry(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<BoundCaptureSelection<'_>, PreparedCaptureSelectionError> {
        self.bind_geometry_impl(geometry, false)
    }

    /// Bind a shortened output allowance while retaining the complete original
    /// prompt axes and chunk schedule. This is semantic geometry only; original
    /// continuation admission must separately prove its saved source and bank.
    pub fn bind_prompt_prefix(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<BoundCaptureSelection<'_>, PreparedCaptureSelectionError> {
        self.bind_geometry_impl(geometry, true)
    }

    fn bind_geometry_impl(
        &self,
        geometry: eredu_core::InferenceGeometry,
        prefix: bool,
    ) -> Result<BoundCaptureSelection<'_>, PreparedCaptureSelectionError> {
        geometry.validate()?;
        let admission = self.source.admission();
        let request = admission.request();
        if request.batch != geometry.batch_size
            || request.prompt_tokens != geometry.input_positions
            || if prefix {
                geometry.max_output_tokens > request.max_predictions
            } else {
                request.max_predictions != geometry.max_output_tokens
            }
            || admission.text_origin().map(|o| o.cached_positions)
                != Some(geometry.cached_positions)
        {
            return Err(PreparedCaptureSelectionError::Identity);
        }
        if self.physical_output(geometry.output) != geometry.output {
            return Err(PreparedCaptureSelectionError::Readout);
        }
        for index in 0..admission.plan().selections.len() {
            if self.declaration(index)?.is_some() {
                if matches!(
                    admission.plan().selections[index].transform,
                    eredu_core::capture::CaptureTransform::RoutedUnits
                ) {
                    eredu_core::capture::CaptureRoutedPrefillPlan::prepare(admission, index, geometry)?;
                } else if matches!(
                    admission.plan().selections[index].transform,
                    eredu_core::capture::CaptureTransform::TopCandidates { .. }
                ) {
                    eredu_core::capture::CaptureCandidateGeometry::prepare(
                        admission,
                        index,
                        eredu_core::capture::CapturePhase::Prefill,
                        0,
                        None,
                    )
                    .map_err(eredu_core::capture::CapturePrefillGeometryError::from)?;
                } else if matches!(
                    admission.plan().selections[index].transform,
                    eredu_core::capture::CaptureTransform::TokenScores { .. }
                ) {
                    eredu_core::capture::CaptureTokenScoreGeometry::prepare(
                        admission,
                        index,
                        eredu_core::capture::CapturePhase::Prefill,
                        0,
                        None,
                    )
                    .map_err(eredu_core::capture::CapturePrefillGeometryError::from)?;
                } else if (self.media
                    || matches!(
                        admission.plan().selections[index].transform,
                        eredu_core::capture::CaptureTransform::Summary
                            | eredu_core::capture::CaptureTransform::Histogram { .. }
                    ))
                    && matches!(
                        admission.plan().selections[index].transform,
                        eredu_core::capture::CaptureTransform::Summary
                            | eredu_core::capture::CaptureTransform::Histogram { .. }
                            | eredu_core::capture::CaptureTransform::Preview { .. }
                    )
                {
                    eredu_core::capture::CapturePrefillTransformPlan::prepare(
                        admission, index, geometry,
                    )?;
                } else {
                    eredu_core::capture::CapturePrefillRowAssembly::prepare(
                        admission, index, geometry,
                    )?;
                }
            }
        }
        Ok(BoundCaptureSelection {
            selection: self,
            geometry,
        })
    }
    /// Physical identity equality, never equal contents or a digest substitute.
    pub fn validate_sources(
        &self,
        source: &eredu_core::capture::SharedCapturePlan,
        paths: &SharedLayeredObservationPaths,
    ) -> Result<(), PreparedCaptureSelectionError> {
        if !self.source.same_storage(source) || !self.paths.same_storage(paths) {
            return Err(PreparedCaptureSelectionError::Identity);
        }
        Ok(())
    }
}
/// Borrow of one retained companion and exact candidate. Not a funded claim,
/// callback/ticket, native permission, completion or causal intervention proof.
#[derive(Debug, Clone, Copy)]
pub struct BoundCaptureSelection<'a> {
    selection: &'a PreparedCaptureSelection,
    geometry: eredu_core::InferenceGeometry,
}
impl<'a> BoundCaptureSelection<'a> {
    /// Exact retained source/declaration association.
    pub fn selection(self) -> &'a PreparedCaptureSelection {
        self.selection
    }
    /// Exact candidate geometry, including actual physical readout.
    pub fn geometry(self) -> eredu_core::InferenceGeometry {
        self.geometry
    }
}
