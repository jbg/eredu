//! Bounded effective parameter access and immutable, versioned edit authority.
//! Plans describe logical loaded parameters, never checkpoint files or native handles.
use crate::{
    capture::{elements, CaptureError, CaptureUsage},
    intervention::InterventionDtype,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod coordination;
mod partition;
pub use coordination::{
    ParameterCoordinationError, ParameterCoordinationStage, ParameterCoordinationUsage,
};
pub use partition::{
    ParameterCoordinateMap, ParameterProjectionFragment, ParameterRegionFragment,
    PartitionParameterRegion,
};

#[cfg(test)]
mod tests;

/// Wire version of parameter plans and results.
pub const PARAMETER_SCHEMA_VERSION: u32 = 1;

/// Selected input arithmetic when a loaded matrix is used as a projection.
/// Weight entries alone do not describe a nonlinear input transformation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProjectionInputTransform {
    /// Older or incomplete discovery provides no verified input-arithmetic fact.
    #[default]
    Unspecified,
    /// The projection consumes its supplied floating input directly.
    Identity,
    /// Independently quantize contiguous feature blocks to E4M3 and multiply by
    /// their F32 scales. Scale is max(max(abs(block)), floor) / magnitude;
    /// values are rounded to the nearest E4M3 value, with ties to even.
    BlockFp8E4m3 {
        /// Last-axis block width; a final partial block has the same rule.
        block_width: u32,
        /// Positive floor applied to the block's maximum absolute input value.
        floor: crate::component::ComponentScalar,
        /// Largest representable positive E4M3 magnitude.
        magnitude: crate::component::ComponentScalar,
    },
}

/// Operations verified for the actual loaded realization of a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ParameterAccess {
    /// Bounded effective entry reads through the selected execution.
    pub query: bool,
    /// Bounded contractions through the selected execution.
    pub projection: bool,
    /// Admitted reversible replacement through the selected execution.
    pub replacement: bool,
}

/// One actual loaded parameter slot and its authoritative shared identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadedParameter {
    /// Architecture-owned canonical slot name.
    pub id: String,
    /// Shared/tied source; editing any alias edits all slots in this class.
    pub shared_id: String,
    /// Logical effective geometry, after materialization transforms.
    pub shape: Vec<u64>,
    /// Exact floating dtype of supported access; absent for unsupported encodings.
    pub dtype: Option<InterventionDtype>,
    /// Whether bounded access and replacement are implemented for this realization.
    pub supported: bool,
    /// Exact operation support. Older records use `supported` for all operations.
    /// When present this takes precedence over that aggregate compatibility field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access: Option<ParameterAccess>,
    /// Concrete implementation or encoding condition.
    pub condition: String,
    /// Actual selected projection input arithmetic, including active overlays.
    #[serde(default)]
    pub input_transform: ProjectionInputTransform,
}
impl LoadedParameter {
    /// Selected operation support, including compatibility with older discovery.
    pub fn access(&self) -> ParameterAccess {
        self.access.unwrap_or(ParameterAccess {
            query: self.supported,
            projection: self.supported,
            replacement: self.supported,
        })
    }
}

/// Loaded execution facts; declarations alone do not authorize native operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterDiscovery {
    /// Current identity includes the loaded session and active parameter version.
    pub identity: String,
    /// Exact prepared artifact identity.
    pub artifact_identity: String,
    /// Active immutable edit identity, if any.
    pub overlay_identity: Option<String>,
    /// Actual loaded slots; unavailable traversal returns a typed unsupported error.
    pub parameters: Vec<LoadedParameter>,
    /// Cumulative reservations; removal and reset never refund these charges.
    pub usage: CaptureUsage,
    /// Separately admitted session-control work, including rejected attempts.
    /// This is never refunded by model or snapshot restoration.
    #[serde(default)]
    pub coordination_usage: ParameterCoordinationUsage,
}

/// Exact rectangular selection in logical parameter coordinates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterRegion {
    /// Start on every axis.
    pub starts: Vec<u64>,
    /// Selected extents, with no implicit broadcast or squeezed axes.
    pub shape: Vec<u64>,
}
impl ParameterRegion {
    /// Checks every extent, including overflow, before any native work.
    pub fn validate(&self, source: &[u64]) -> Result<u64, ParameterError> {
        if self.starts.len() != source.len() || self.shape.len() != source.len() {
            return Err(ParameterError::Invalid(
                "region rank differs from parameter".into(),
            ));
        }
        for ((start, width), source) in self.starts.iter().zip(&self.shape).zip(source) {
            if *width == 0 || start.checked_add(*width).ok_or(ParameterError::Overflow)? > *source {
                return Err(ParameterError::Invalid(
                    "empty or out-of-bounds parameter region".into(),
                ));
            }
        }
        elements(&self.shape).map_err(Into::into)
    }
    fn overlaps(&self, other: &Self) -> bool {
        self.starts
            .iter()
            .zip(&self.shape)
            .zip(other.starts.iter().zip(&other.shape))
            .all(|((a, n), (b, m))| *a < b + m && *b < a + n)
    }
}

/// A finite host update, evaluated in F32 and rounded once to the exact target dtype.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParameterUpdate {
    /// Replaces selected effective entries.
    Replace {
        /// Row-major values.
        values: Vec<f32>,
    },
    /// Adds to selected effective entries.
    Add {
        /// Row-major deltas.
        values: Vec<f32>,
    },
}
impl ParameterUpdate {
    /// Bounded row-major host values.
    pub fn values(&self) -> &[f32] {
        match self {
            Self::Replace { values } | Self::Add { values } => values,
        }
    }
}

/// One exact logical edit. Disjoint edits may target the same shared parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterEdit {
    /// Caller operation identity.
    pub id: String,
    /// Canonical slot or alias from loaded discovery.
    pub parameter: String,
    /// Exact full logical shape, binding the intended geometry.
    pub parameter_shape: Vec<u64>,
    /// Exact target dtype, with no implicit dtype inference.
    pub dtype: InterventionDtype,
    /// Selected read row, write column or other exact rectangle.
    pub region: ParameterRegion,
    /// Requested update.
    pub update: ParameterUpdate,
}

/// Serializable multi-parameter transaction. It is not executable authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterOverlayPlan {
    /// Supported wire version.
    pub schema_version: u32,
    /// Loaded discovery identity against which values were measured.
    pub base_identity: String,
    /// Consumer-supplied provenance, included in the edit digest.
    pub provenance: String,
    /// Ordered operations. Overlapping regions in one shared class are rejected.
    pub edits: Vec<ParameterEdit>,
}

/// Immutable authority admitted for one loaded execution and base parameter version.
#[derive(Debug, Clone)]
pub struct AdmittedParameterOverlay {
    plan: ParameterOverlayPlan,
    identity: String,
    intent_identity: String,
    artifact_identity: String,
    prior_overlay: Option<String>,
    shared_targets: Vec<String>,
}
impl AdmittedParameterOverlay {
    /// Validates topology, exact geometry/dtype, sharing, finite values and plan bounds.
    pub fn admit(
        plan: ParameterOverlayPlan,
        discovery: &ParameterDiscovery,
    ) -> Result<Self, ParameterError> {
        if plan.schema_version != PARAMETER_SCHEMA_VERSION
            || plan.base_identity != discovery.identity
        {
            return Err(ParameterError::StaleIdentity);
        }
        if plan.edits.is_empty() || plan.edits.len() > 1024 || plan.provenance.len() > 4096 {
            return Err(ParameterError::Invalid(
                "overlay requires 1..=1024 edits and bounded provenance".into(),
            ));
        }
        let mut ids = std::collections::BTreeSet::new();
        let mut shared_targets = vec![];
        let mut payload = 0u64;
        for (index, edit) in plan.edits.iter().enumerate() {
            if edit.id.is_empty() || edit.id.len() > 256 || !ids.insert(&edit.id) {
                return Err(ParameterError::Invalid(
                    "empty, duplicate or oversized edit identity".into(),
                ));
            }
            let parameter = discovery
                .parameters
                .iter()
                .find(|p| p.id == edit.parameter)
                .ok_or_else(|| ParameterError::Missing(edit.parameter.clone()))?;
            if !parameter.access().replacement {
                return Err(ParameterError::Unsupported(parameter.condition.clone()));
            }
            if edit.parameter_shape != parameter.shape || Some(edit.dtype) != parameter.dtype {
                return Err(ParameterError::Invalid(
                    "parameter shape or dtype differs from loaded execution".into(),
                ));
            }
            let count = edit.region.validate(&parameter.shape)?;
            if count != edit.update.values().len() as u64
                || edit.update.values().iter().any(|v| !v.is_finite())
            {
                return Err(ParameterError::Invalid(
                    "edit payload shape or finite-value constraint".into(),
                ));
            }
            payload = payload
                .checked_add(count.checked_mul(4).ok_or(ParameterError::Overflow)?)
                .ok_or(ParameterError::Overflow)?;
            if payload > 16 << 20 {
                return Err(ParameterError::Invalid(
                    "overlay host payload exceeds 16 MiB".into(),
                ));
            }
            for (prior, target) in plan.edits[..index].iter().zip(&shared_targets) {
                if target == &parameter.shared_id && edit.region.overlaps(&prior.region) {
                    return Err(ParameterError::Conflict(parameter.shared_id.clone()));
                }
            }
            shared_targets.push(parameter.shared_id.clone());
        }
        // Preserve the local admitted identity while comparing identical global
        // edits across independently realized rank-local sessions. Stream both
        // digests; admission never creates a second full JSON payload buffer.
        let identity = parameter_digest("overlay", &plan)?;
        let intent_identity = parameter_digest(
            "parameter-intent",
            &(
                plan.schema_version,
                &discovery.artifact_identity,
                &discovery.overlay_identity,
                &plan.provenance,
                &plan.edits,
                &shared_targets,
            ),
        )?;
        Ok(Self {
            plan,
            identity,
            intent_identity,
            artifact_identity: discovery.artifact_identity.clone(),
            prior_overlay: discovery.overlay_identity.clone(),
            shared_targets,
        })
    }
    /// Exact immutable admitted host plan.
    pub fn plan(&self) -> &ParameterOverlayPlan {
        &self.plan
    }
    /// Content digest including source/execution authority and provenance.
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Exact ordered edits, sharing, source and prior overlay, excluding the
    /// rank-local loaded identity. Peers compare this descriptive intent while
    /// retaining their own admitted authority. It grants no setup, branch,
    /// parameter-version, native publication or communication authority.
    pub fn intent_identity(&self) -> &str {
        &self.intent_identity
    }
    /// Canonical sharing class for each operation, in the admitted order.
    pub fn shared_targets(&self) -> &[String] {
        &self.shared_targets
    }
    /// Revalidates against actual current execution; serialized plans cannot forge authority.
    pub fn validate(&self, discovery: &ParameterDiscovery) -> Result<(), ParameterError> {
        if self.plan.base_identity != discovery.identity
            || self.artifact_identity != discovery.artifact_identity
            || self.prior_overlay != discovery.overlay_identity
        {
            return Err(ParameterError::StaleIdentity);
        }
        // The admitted plan is private and immutable. Recheck only the loaded authority;
        // reserializing or cloning the payload here would perform unnecessary host work.
        for (edit, shared) in self.plan.edits.iter().zip(&self.shared_targets) {
            let parameter = discovery
                .parameters
                .iter()
                .find(|p| p.id == edit.parameter)
                .ok_or_else(|| ParameterError::Missing(edit.parameter.clone()))?;
            if !parameter.access().replacement {
                return Err(ParameterError::Unsupported(parameter.condition.clone()));
            }
            if parameter.shape != edit.parameter_shape
                || parameter.dtype != Some(edit.dtype)
                || &parameter.shared_id != shared
            {
                return Err(ParameterError::StaleIdentity);
            }
        }
        Ok(())
    }
}

fn parameter_digest(prefix: &str, value: &impl Serialize) -> Result<String, ParameterError> {
    struct Writer {
        digest: Sha256,
        bytes: u64,
    }
    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes = self
                .bytes
                .checked_add(bytes.len() as u64)
                .filter(|size| *size <= 128 << 20)
                .ok_or_else(|| std::io::Error::other("encoded overlay exceeds 128 MiB"))?;
            self.digest.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Writer {
        digest: Sha256::new(),
        bytes: 0,
    };
    serde_json::to_writer(&mut writer, value)
        .map_err(|error| ParameterError::Invalid(error.to_string()))?;
    let digest = writer
        .digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("{prefix}-{digest}"))
}

/// Contracts an effective parameter region along one logical axis with bounded host directions.
/// Directions are row-major `[directions, region.shape[axis]]`. Output axes retain their
/// original order with the contracted axis removed, followed by the direction axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterProjection {
    /// Logical rectangle to project, without first copying its weights to the host.
    pub region: ParameterRegion,
    /// Axis contracted with each direction.
    pub axis: usize,
    /// Number of independent directions (including token differences or read covectors).
    pub directions: u64,
    /// Finite binary32 directions; accumulation uses the backend's F32 matrix multiplication.
    pub coefficients: Vec<f32>,
}
impl ParameterProjection {
    /// Validates bounded input and checked output geometry before native work.
    pub fn output_shape(&self, source: &[u64]) -> Result<Vec<u64>, ParameterError> {
        self.region.validate(source)?;
        let width =
            *self.region.shape.get(self.axis).ok_or_else(|| {
                ParameterError::Invalid("projection axis is out of bounds".into())
            })?;
        let count = width
            .checked_mul(self.directions)
            .ok_or(ParameterError::Overflow)?;
        if self.directions == 0
            || count > (16 << 20) / 4
            || count != self.coefficients.len() as u64
            || self.coefficients.iter().any(|x| !x.is_finite())
        {
            return Err(ParameterError::Invalid(
                "projection directions require a finite, exact payload of at most 16 MiB".into(),
            ));
        }
        let mut shape: Vec<_> = self
            .region
            .shape
            .iter()
            .enumerate()
            .filter_map(|(axis, n)| (axis != self.axis).then_some(*n))
            .collect();
        shape.push(self.directions);
        elements(&shape)?;
        Ok(shape)
    }
}

/// Bounded native projection of effective weights, including any active overlay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterProjectionValues {
    /// Actual loaded execution and parameter version.
    pub identity: String,
    /// Requested effective parameter.
    pub parameter: String,
    /// Original parameter dtype; output values are F32.
    pub source_dtype: InterventionDtype,
    /// Explicit output geometry with the direction axis last.
    pub shape: Vec<u64>,
    /// Finite row-major native reduction values.
    pub values: Vec<f32>,
    /// Cumulative accounting, which includes earlier failed work.
    pub usage: CaptureUsage,
}

/// Finite, bounded F32 host values of one effective parameter region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterValues {
    /// Execution and active edit version actually read.
    pub identity: String,
    /// Canonical requested slot.
    pub parameter: String,
    /// Exact source dtype, widened without changing its represented values.
    pub dtype: InterventionDtype,
    /// Explicit region and shape.
    pub region: ParameterRegion,
    /// Row-major finite values.
    pub values: Vec<f32>,
    /// Updated cumulative accounting, including failed earlier operations.
    pub usage: CaptureUsage,
}

/// Typed failure for access, admission and atomic parameter publication.
#[derive(Debug, thiserror::Error)]
pub enum ParameterError {
    /// Distributed agreement, completion or peer rejection.
    #[error(transparent)]
    Coordination(Box<ParameterCoordinationError>),
    /// Invalid geometry, dtype, payload or request.
    #[error("invalid parameter operation: {0}")]
    Invalid(String),
    /// Actual realization lacks the requested mechanism.
    #[error("parameter operation unsupported: {0}")]
    Unsupported(String),
    /// Canonical parameter was not found.
    #[error("missing loaded parameter: {0}")]
    Missing(String),
    /// The selected global value lacks complete, valid producer coverage.
    #[error("incomplete parameter result: {0}")]
    Incomplete(String),
    /// Authority belongs to another loaded execution or parameter version.
    #[error("stale parameter execution identity")]
    StaleIdentity,
    /// Two operations overlap after canonical alias resolution.
    #[error("overlapping edits of shared parameter: {0}")]
    Conflict(String),
    /// Checked size arithmetic overflowed.
    #[error("parameter size arithmetic overflow")]
    Overflow,
    /// Logical storage, host or encoded reservation failed before native work.
    #[error(transparent)]
    Budget(#[from] CaptureError),
    /// Native cause preserved behind the neutral application boundary.
    #[error(transparent)]
    Backend(#[from] crate::BackendFailure),
}

impl From<ParameterCoordinationError> for ParameterError {
    fn from(error: ParameterCoordinationError) -> Self {
        Self::Coordination(Box::new(error))
    }
}

/// Additive backend mechanism implemented by loaded parameter access providers.
pub trait ParameterBackend: crate::TextGenerationBackend {
    /// Reads actual loaded geometry without evaluating or retaining parameter
    /// tensors. Distributed implementations may exchange bounded prepared
    /// metadata through the retained communicator; every rank participates.
    fn parameter_discovery(
        runtime: &mut crate::ModelRuntime<Self>,
    ) -> Result<ParameterDiscovery, ParameterError>;
    /// Copies only the admitted region after reserving cumulative resources.
    fn query_parameter(
        runtime: &mut crate::ModelRuntime<Self>,
        identity: &str,
        parameter: &str,
        region: ParameterRegion,
        limits: CaptureUsage,
    ) -> Result<ParameterValues, ParameterError>;
    /// Projects effective weights natively, copying only bounded reduction output to the host.
    fn project_parameter(
        runtime: &mut crate::ModelRuntime<Self>,
        identity: &str,
        parameter: &str,
        projection: ParameterProjection,
        limits: CaptureUsage,
    ) -> Result<ParameterProjectionValues, ParameterError>;
    /// Prepares every replacement before one atomic publication at an idle boundary.
    fn activate_parameter_overlay(
        runtime: &mut crate::ModelRuntime<Self>,
        overlay: &AdmittedParameterOverlay,
        limits: CaptureUsage,
    ) -> Result<ParameterDiscovery, ParameterError>;
    /// Restores original parameters and invalidates incompatible mutable state.
    fn remove_parameter_overlay(
        runtime: &mut crate::ModelRuntime<Self>,
        identity: &str,
    ) -> Result<ParameterDiscovery, ParameterError>;
}
