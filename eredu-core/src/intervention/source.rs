//! Closed copies of immutable admitted edits; no execution or budget admission.
use super::*;
use crate::{HostPreparationAuthority, capture::plan_copy::Worker};
use std::{alloc::Layout, mem::size_of, sync::Arc};
mod copy;

/// Fixed source-validation failure. No error formatting or source clone occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InterventionSourceError {
    /// Actual loaded source, selected declarations or session changed.
    #[error("intervention does not belong to this loaded source and session")]
    Identity,
    /// Canonical closed-plan serialization failed.
    #[error("intervention identity serialization failed")]
    Encoding,
}
struct Inner {
    plan: AdmittedInterventionPlan,
    evidence: Vec<Option<InterventionEvidenceCompanion>>,
    _host: HostPreparationAuthority,
}
/// Immutable copied plan. Its closed Arc retires before the copied payload and
/// finally the actual caller's host authority. It cannot expose a mutable plan.
#[derive(Clone)]
pub struct SharedInterventionPlan(Option<Arc<Inner>>);
impl std::fmt::Debug for SharedInterventionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedInterventionPlan")
            .field("plan", self.admission())
            .finish_non_exhaustive()
    }
}
impl Drop for SharedInterventionPlan {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl SharedInterventionPlan {
    /// Borrow the exact immutable admission; caller clones are separate storage.
    pub fn admission(&self) -> &AdmittedInterventionPlan {
        &self.0.as_ref().expect("live intervention source").plan
    }
    /// Borrow immutable evidence declarations belonging to this exact operation.
    /// Their zero budgets grant no capture quota or independent execution.
    pub fn evidence(&self, operation: usize) -> Option<&InterventionEvidenceCompanion> {
        self.0
            .as_ref()
            .expect("live intervention source")
            .evidence
            .get(operation)?
            .as_ref()
    }
    /// Physical source identity; a matching semantic digest is insufficient.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    fn control_bytes() -> Option<usize> {
        Some(
            Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
                .extend(Layout::new::<Inner>())
                .ok()?
                .0
                .pad_to_align()
                .size()
                .checked_add(size_of::<Self>() + size_of::<Option<Inner>>())?,
        )
    }
}
/// Exact borrowed count/copy program. A caller must pay its full extent before
/// copying; the supplied host token is custody and grants no admission itself.
#[derive(Debug)]
pub struct PreparedInterventionPlanCopy<'a> {
    source: &'a AdmittedInterventionPlan,
    bytes: usize,
}
impl<'a> PreparedInterventionPlanCopy<'a> {
    fn controls() -> Option<usize> {
        [
            size_of::<Worker>(),
            size_of::<Self>(),
            size_of::<AdmittedInterventionPlan>() * 2,
            size_of::<HostPreparationAuthority>(),
            size_of::<CapturePlanCopyError>(),
            size_of::<Result<SharedInterventionPlan, CapturePlanCopyError>>(),
            SharedInterventionPlan::control_bytes()?,
            InterventionEvidenceCompanion::control_bytes()?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Count the same typed DTO copy without allocating any destination.
    pub fn inspect(source: &'a AdmittedInterventionPlan) -> Result<Self, CapturePlanCopyError> {
        let mut worker = Worker::with_controls(
            false,
            Self::controls().ok_or(CapturePlanCopyError::Overflow)?,
        );
        drop(copy::plan(&mut worker, source)?);
        drop(copy::evidence(&mut worker, source)?);
        Ok(Self {
            source,
            bytes: worker.bytes(),
        })
    }
    /// Source-derived payload, shared shell and concrete construction controls.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Fixed nonallocating inspection stack, paid by the preparation caller.
    pub fn inspection_control_bytes() -> Option<usize> {
        [
            size_of::<Worker>(),
            size_of::<Self>(),
            size_of::<AdmittedInterventionPlan>(),
            size_of::<InterventionOperation>(),
            size_of::<InterventionPoint>(),
            size_of::<InterventionAction>(),
            size_of::<InterventionTensor>(),
            size_of::<TensorAxis>(),
            InterventionEvidenceCompanion::control_bytes()?,
            size_of::<Result<Self, CapturePlanCopyError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|n| n.checked_mul(2))
    }
    /// Fresh copy only. Existing caller strings, vectors and payloads are never
    /// moved into retained custody or readmitted as a replacement execution.
    pub fn copy(
        self,
        host: HostPreparationAuthority,
    ) -> Result<SharedInterventionPlan, CapturePlanCopyError> {
        let mut worker = Worker::with_controls(
            true,
            Self::controls().ok_or(CapturePlanCopyError::Overflow)?,
        );
        let plan = copy::plan(&mut worker, self.source)?;
        let mut evidence = copy::evidence(&mut worker, self.source)?;
        if worker.bytes() != self.bytes {
            return Err(CapturePlanCopyError::Capacity);
        }
        // Move each already copied geometry into its counted shared shell.
        // No DTO/string/vector copy or replacement admission occurs here. Every
        // escaping evidence alias retains this same original host authority.
        for companion in &mut evidence {
            if let Some(value) = companion.take() {
                *companion = Some(value.into_shared(host.clone()));
            }
        }
        Ok(SharedInterventionPlan(Some(Arc::new(Inner {
            plan,
            evidence,
            _host: host,
        }))))
    }
}

impl AdmittedInterventionPlan {
    /// Compare an immutable prior admission with the actual loaded discovery.
    /// Reuses its canonical digest worker without cloning, readmitting or
    /// allocating replacement plan/point/string destinations.
    pub fn validate_discovery(
        &self,
        discovery: &InterventionDiscovery,
    ) -> Result<(), InterventionSourceError> {
        if discovery.schema_version != INTERVENTION_SCHEMA_VERSION
            || discovery.artifact_identity != self.artifact_identity
            || discovery
                .session_identity
                .as_ref()
                .is_none_or(String::is_empty)
            || self.points.len() != self.plan.operations.len()
        {
            return Err(InterventionSourceError::Identity);
        }
        for (operation, point) in self.plan.operations.iter().zip(&self.points) {
            let mut found = discovery
                .points
                .iter()
                .filter(|candidate| candidate.path == operation.target);
            if found.next() != Some(point) || found.next().is_some() {
                return Err(InterventionSourceError::Identity);
            }
        }
        self.validate_identity_parts(&discovery.artifact_identity,
            discovery.session_identity.as_deref().ok_or(InterventionSourceError::Identity)?)
    }
    /// Authenticate only the source/session part of an admission using actual
    /// resolved content identity. The owner must separately compare the selected
    /// declarations through the same loaded support projection.
    pub fn validate_source_identity(&self,artifact:crate::artifact::ArtifactIdentity,session:&str)
        ->Result<(),InterventionSourceError> {
        let mut encoded=[0u8;7+64];
        encoded[..7].copy_from_slice(b"sha256:");
        write_hex(&mut encoded[7..],&artifact.digest());
        self.validate_identity_parts(std::str::from_utf8(&encoded).expect("ASCII identity"),session)
    }
    fn validate_identity_parts(&self,artifact:&str,session:&str)->Result<(),InterventionSourceError> {
        if artifact!=self.artifact_identity || session.is_empty() || self.points.len()!=self.plan.operations.len() {
            return Err(InterventionSourceError::Identity);
        }
        let digest = intervention_digest_bytes(&(
            &self.plan,
            &self.points,
            self.request,
            artifact,
            Some(session),
            &self.session_id,
        ))
        .map_err(|_| InterventionSourceError::Encoding)?;
        let mut encoded = [0u8; IDENTITY_PREFIX.len() + 1 + 64];
        encoded[..IDENTITY_PREFIX.len()].copy_from_slice(IDENTITY_PREFIX.as_bytes());
        encoded[IDENTITY_PREFIX.len()] = b'-';
        write_hex(&mut encoded[IDENTITY_PREFIX.len() + 1..], &digest);
        if let Some(bounds) = self.invocation_bounds {
            let identity = std::str::from_utf8(&encoded).expect("ASCII digest");
            let digest = intervention_digest_bytes(&("invocation", identity, bounds))
                .map_err(|_| InterventionSourceError::Encoding)?;
            write_hex(&mut encoded[IDENTITY_PREFIX.len() + 1..], &digest);
        } else if self.text_origin != CaptureTextOrigin::default() {
            let identity = std::str::from_utf8(&encoded).expect("ASCII digest");
            let digest = intervention_digest_bytes(&("text_origin", identity, self.text_origin))
                .map_err(|_| InterventionSourceError::Encoding)?;
            write_hex(&mut encoded[IDENTITY_PREFIX.len() + 1..], &digest);
        }
        if encoded.as_slice() != self.identity.as_bytes() {
            return Err(InterventionSourceError::Identity);
        }
        Ok(())
    }
    /// Exact fixed borrowed-validation representations. No native/source grant.
    pub fn discovery_validation_control_bytes() -> Option<usize> {
        [
            size_of::<[u8;7+64]>(),
            size_of::<crate::artifact::ArtifactIdentity>(),
            size_of::<Option<&str>>(),
            size_of::<DigestWriter>(),
            size_of::<serde_json::Serializer<&mut DigestWriter>>(),
            size_of::<[u8; 32]>(),
            size_of::<[u8; IDENTITY_PREFIX.len() + 1 + 64]>(),
            size_of::<Result<[u8; 32], serde_json::Error>>(),
            size_of::<Result<(), InterventionSourceError>>(),
            size_of::<(
                &InterventionPlan,
                &Vec<InterventionPoint>,
                CaptureRequestShape,
                &String,
                &Option<String>,
                &String,
            )>(),
            size_of::<(&str, &str, CaptureInvocationBounds)>(),
            size_of::<(&str, &str, CaptureTextOrigin)>(),
            size_of::<(&Self, &InterventionDiscovery)>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}
fn write_hex(out: &mut [u8], digest: &[u8; 32]) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (out, &byte) in out.chunks_exact_mut(2).zip(digest) {
        out[0] = HEX[(byte >> 4) as usize];
        out[1] = HEX[(byte & 15) as usize];
    }
}
