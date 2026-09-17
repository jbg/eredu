//! Borrowed loaded identity validation without rebuilding an admission DTO.
use super::AdmittedSpeculativeActivations;
use crate::artifact::ArtifactIdentity;
use std::{
    io,
    mem::{size_of, size_of_val},
};

/// Fixed loaded-source mismatch. It allocates no diagnostic or replacement plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpeculativeActivationSourceError {
    /// Artifact, selected execution, overlay or realized session differs.
    #[error("speculative activations do not belong to this loaded execution and session")]
    Identity,
    /// A retained architecture declaration or selected scope differs.
    #[error("speculative activation declaration differs from the selected architecture")]
    Declaration,
    /// Internal edits have no original invocation producer in this profile.
    #[error("original internal speculative intervention production is not implemented")]
    UnqualifiedIntervention,
    /// The closed source identity could not be serialized.
    #[error("speculative activation source identity serialization failed")]
    Encoding,
}
struct Comparison<'a> {
    remaining: &'a [u8],
    equal: bool,
}
impl io::Write for Comparison<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.equal &= self.remaining.starts_with(bytes);
        self.remaining = self.remaining.get(bytes.len()..).unwrap_or_default();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl AdmittedSpeculativeActivations {
    /// Compare exact loaded identity fields using the same serialized execution
    /// and overlay tuple as architecture discovery. This grants no invocation,
    /// capture budget or collector capability. The loaded owner must separately
    /// compare retained points, scopes and actual selected mechanism facts.
    pub fn validate_source_identity(
        &self,
        artifact: ArtifactIdentity,
        execution: &str,
        overlay: Option<&str>,
        session: &str,
    ) -> Result<(), SpeculativeActivationSourceError> {
        let mut encoded = [0u8; 71];
        encoded[..7].copy_from_slice(b"sha256:");
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for (index, byte) in artifact.digest().into_iter().enumerate() {
            encoded[7 + index * 2] = HEX[usize::from(byte >> 4)];
            encoded[8 + index * 2] = HEX[usize::from(byte & 15)];
        }
        if self.artifact_identity.as_bytes() != encoded
            || session.is_empty()
            || self.session_identity != session
        {
            return Err(SpeculativeActivationSourceError::Identity);
        }
        let mut comparison = Comparison {
            remaining: self.execution_identity.as_bytes(),
            equal: true,
        };
        serde_json::to_writer(&mut comparison, &(execution, overlay))
            .map_err(|_| SpeculativeActivationSourceError::Encoding)?;
        if !comparison.equal || !comparison.remaining.is_empty() {
            return Err(SpeculativeActivationSourceError::Identity);
        }
        Ok(())
    }
    /// Fixed borrowed-comparison frames, including the real serializer/writer.
    /// No source payload or native execution authority is represented here.
    pub fn source_validation_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Comparison<'_>>(),
            size_of::<[u8; 71]>(),
            size_of::<[u8; 32]>(),
            size_of::<ArtifactIdentity>(),
            size_of::<serde_json::Serializer<&mut Comparison<'_>>>(),
            size_of::<Result<(), serde_json::Error>>(),
            size_of::<Result<(), SpeculativeActivationSourceError>>(),
            size_of::<(&str, Option<&str>)>(),
            size_of::<(&Self, ArtifactIdentity, &str, Option<&str>, &str)>(),
            size_of::<std::iter::Enumerate<std::array::IntoIter<u8, 32>>>(),
            size_of::<io::Result<usize>>(),
            size_of::<io::Result<()>>(),
            size_of::<(&mut Comparison<'_>, &[u8])>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
