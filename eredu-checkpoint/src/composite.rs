//! Family-neutral schemas for multi-artifact checkpoints.

use std::collections::{BTreeMap, BTreeSet};

use crate::WeightQuantization;

/// Stable component identity within a composite checkpoint.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComponentId(String);

impl ComponentId {
    /// Creates a non-empty component identity.
    pub fn new(value: impl Into<String>) -> Result<Self, CompositeArtifactError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(CompositeArtifactError::EmptyComponent);
        }
        Ok(Self(value))
    }

    /// Returns the stable component name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Semantic role of one independently opened checkpoint artifact.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ArtifactRole {
    /// Primary decoder artifact.
    Decoder,
    /// Vision-tower artifact.
    Vision,
    /// Audio-tower artifact.
    Audio,
    /// Media-to-decoder projection artifact.
    Projector,
    /// External speculative assistant artifact.
    Assistant,
    /// Architecture-declared role not represented by a common variant.
    Named(String),
}

/// One artifact admitted by a composite model checkpoint.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArtifactComponentSchema {
    /// Stable component identity.
    pub component: ComponentId,
    /// Semantic artifact role.
    pub role: ArtifactRole,
    /// Whether construction fails when the artifact is absent.
    pub required: bool,
    /// Architecture identity expected in the artifact header.
    pub architecture: String,
}

/// Primary decoder plus optional or required sibling artifacts.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompositeArtifactSchema {
    primary: ArtifactComponentSchema,
    siblings: Vec<ArtifactComponentSchema>,
}

impl CompositeArtifactSchema {
    /// Validates a primary decoder and its sibling artifact declarations.
    pub fn new(
        primary: ArtifactComponentSchema,
        siblings: impl IntoIterator<Item = ArtifactComponentSchema>,
    ) -> Result<Self, CompositeArtifactError> {
        if primary.role != ArtifactRole::Decoder {
            return Err(CompositeArtifactError::PrimaryNotDecoder);
        }
        validate_artifact(&primary)?;
        let siblings = siblings.into_iter().collect::<Vec<_>>();
        let mut components = BTreeSet::from([primary.component.clone()]);
        let mut roles = BTreeSet::from([ArtifactRole::Decoder]);
        for sibling in &siblings {
            validate_artifact(sibling)?;
            if sibling.role == ArtifactRole::Decoder {
                return Err(CompositeArtifactError::DuplicateRole(ArtifactRole::Decoder));
            }
            if !components.insert(sibling.component.clone()) {
                return Err(CompositeArtifactError::DuplicateComponent(
                    sibling.component.clone(),
                ));
            }
            if !matches!(sibling.role, ArtifactRole::Named(_))
                && !roles.insert(sibling.role.clone())
            {
                return Err(CompositeArtifactError::DuplicateRole(sibling.role.clone()));
            }
        }
        Ok(Self { primary, siblings })
    }

    /// Returns the required primary decoder declaration.
    pub fn primary(&self) -> &ArtifactComponentSchema {
        &self.primary
    }

    /// Returns sibling declarations in architecture order.
    pub fn siblings(&self) -> &[ArtifactComponentSchema] {
        &self.siblings
    }

    /// Returns all required component identities.
    pub fn required_components(&self) -> impl Iterator<Item = &ComponentId> {
        std::iter::once(&self.primary)
            .chain(&self.siblings)
            .filter(|artifact| artifact.required)
            .map(|artifact| &artifact.component)
    }
}

fn validate_artifact(artifact: &ArtifactComponentSchema) -> Result<(), CompositeArtifactError> {
    if artifact.architecture.trim().is_empty() {
        return Err(CompositeArtifactError::EmptyArchitecture(
            artifact.component.clone(),
        ));
    }
    if matches!(&artifact.role, ArtifactRole::Named(name) if name.trim().is_empty()) {
        return Err(CompositeArtifactError::EmptyRole);
    }
    Ok(())
}

/// Component-owned logical parameter identities with collision detection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ComponentParameterCatalog {
    owners: BTreeMap<String, ComponentId>,
}

impl ComponentParameterCatalog {
    /// Builds one catalog and rejects cross-component identity collisions.
    pub fn new(
        components: impl IntoIterator<Item = (ComponentId, Vec<String>)>,
    ) -> Result<Self, CompositeArtifactError> {
        let mut owners = BTreeMap::new();
        let mut declared_components = BTreeSet::new();
        for (component, parameters) in components {
            if !declared_components.insert(component.clone()) {
                return Err(CompositeArtifactError::DuplicateComponent(component));
            }
            let mut local = BTreeSet::new();
            for parameter in parameters {
                if parameter.trim().is_empty() {
                    return Err(CompositeArtifactError::EmptyParameter(component.clone()));
                }
                if !local.insert(parameter.clone()) {
                    return Err(CompositeArtifactError::DuplicateParameter {
                        parameter,
                        first: component.clone(),
                        second: component,
                    });
                }
                if let Some(first) = owners.insert(parameter.clone(), component.clone()) {
                    return Err(CompositeArtifactError::DuplicateParameter {
                        parameter,
                        first,
                        second: component,
                    });
                }
            }
        }
        Ok(Self { owners })
    }

    /// Returns the component that owns one logical parameter identity.
    pub fn owner(&self, parameter: &str) -> Option<&ComponentId> {
        self.owners.get(parameter)
    }

    /// Returns all logical parameters in stable sorted order.
    pub fn parameters(&self) -> impl Iterator<Item = (&str, &ComponentId)> {
        self.owners
            .iter()
            .map(|(parameter, component)| (parameter.as_str(), component))
    }
}

/// Canonical convolution kernel dimensionality and physical axis layout.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ConvolutionKernelLayout {
    /// One-dimensional `[output, kernel, input]` kernel.
    OneDimensional,
    /// Two-dimensional `[output, height, width, input]` kernel.
    TwoDimensional,
    /// Three-dimensional `[output, channels, temporal, height, width]` kernel
    /// consumed as a flattened patch projection.
    FlattenedThreeDimensional,
}

impl ConvolutionKernelLayout {
    /// Validates and returns the canonical flattened matrix shape.
    pub fn flattened_shape(self, shape: &[usize]) -> Result<[usize; 2], CompositeArtifactError> {
        let expected_rank = match self {
            Self::OneDimensional => 3,
            Self::TwoDimensional => 4,
            Self::FlattenedThreeDimensional => 5,
        };
        if shape.len() != expected_rank || shape.contains(&0) {
            return Err(CompositeArtifactError::InvalidConvolutionShape {
                layout: self,
                shape: shape.to_vec(),
            });
        }
        let input = shape[1..].iter().try_fold(1_usize, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or(CompositeArtifactError::ShapeOverflow)
        })?;
        Ok([shape[0], input])
    }
}

/// Per-component default and per-weight quantization policy.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ComponentQuantizationSchema {
    /// Default encoding for weights without an override.
    pub default: Option<WeightQuantization>,
    /// Exact logical-weight overrides.
    pub weights: BTreeMap<String, WeightQuantization>,
}

impl ComponentQuantizationSchema {
    /// Validates non-empty weight identities.
    pub fn new(
        default: Option<WeightQuantization>,
        weights: BTreeMap<String, WeightQuantization>,
    ) -> Result<Self, CompositeArtifactError> {
        if weights.keys().any(|name| name.trim().is_empty()) {
            return Err(CompositeArtifactError::EmptyQuantizedWeight);
        }
        Ok(Self { default, weights })
    }

    /// Resolves the selected encoding for one logical weight.
    pub fn quantization(&self, weight: &str) -> Option<WeightQuantization> {
        self.weights.get(weight).copied().or(self.default)
    }
}

/// Compatibility contract checked before opening a sibling projector.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ProjectorCompatibility {
    /// Primary decoder architecture identity.
    pub decoder_architecture: String,
    /// Architecture identity declared by the projector.
    pub projector_architecture: String,
    /// Decoder hidden width.
    pub decoder_hidden_width: usize,
    /// Projector output width.
    pub projector_output_width: usize,
    /// Modality placeholder identities admitted by the decoder.
    pub decoder_modality_tokens: BTreeSet<u32>,
    /// Modality placeholder identities declared by the projector.
    pub projector_modality_tokens: BTreeSet<u32>,
}

impl ProjectorCompatibility {
    /// Rejects architecture, width, and modality-token mismatches atomically.
    pub fn validate(&self) -> Result<(), CompositeArtifactError> {
        if self.decoder_architecture.trim().is_empty()
            || self.projector_architecture.trim().is_empty()
            || self.decoder_architecture != self.projector_architecture
        {
            return Err(CompositeArtifactError::ProjectorArchitectureMismatch);
        }
        if self.decoder_hidden_width == 0
            || self.decoder_hidden_width != self.projector_output_width
        {
            return Err(CompositeArtifactError::ProjectorWidthMismatch {
                decoder: self.decoder_hidden_width,
                projector: self.projector_output_width,
            });
        }
        if self.decoder_modality_tokens != self.projector_modality_tokens {
            return Err(CompositeArtifactError::ProjectorModalityMismatch);
        }
        Ok(())
    }
}

/// Invalid composite checkpoint declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CompositeArtifactError {
    /// A component identity was empty.
    #[error("component identity cannot be empty")]
    EmptyComponent,
    /// The primary artifact was not the decoder.
    #[error("primary composite artifact must have decoder role")]
    PrimaryNotDecoder,
    /// One component was declared more than once.
    #[error("component {0:?} was declared more than once")]
    DuplicateComponent(ComponentId),
    /// One singleton role was declared more than once.
    #[error("artifact role {0:?} was declared more than once")]
    DuplicateRole(ArtifactRole),
    /// One artifact had no architecture identity.
    #[error("component {0:?} has an empty architecture identity")]
    EmptyArchitecture(ComponentId),
    /// A named artifact role was empty.
    #[error("named artifact role cannot be empty")]
    EmptyRole,
    /// A logical parameter identity was empty.
    #[error("component {0:?} contains an empty parameter identity")]
    EmptyParameter(ComponentId),
    /// Two declarations own the same logical parameter identity.
    #[error("parameter {parameter:?} is owned by both {first:?} and {second:?}")]
    DuplicateParameter {
        /// Colliding logical parameter.
        parameter: String,
        /// First owner.
        first: ComponentId,
        /// Second owner.
        second: ComponentId,
    },
    /// Convolution shape did not match its declared physical layout.
    #[error("invalid {layout:?} convolution shape {shape:?}")]
    InvalidConvolutionShape {
        /// Declared layout.
        layout: ConvolutionKernelLayout,
        /// Invalid physical shape.
        shape: Vec<usize>,
    },
    /// Shape arithmetic overflowed.
    #[error("checkpoint shape arithmetic overflowed")]
    ShapeOverflow,
    /// A per-weight quantization key was empty.
    #[error("quantized weight identity cannot be empty")]
    EmptyQuantizedWeight,
    /// Projector architecture identity differs from the decoder.
    #[error("projector architecture does not match the decoder")]
    ProjectorArchitectureMismatch,
    /// Projector output width differs from the decoder hidden width.
    #[error("projector output width {projector} does not match decoder hidden width {decoder}")]
    ProjectorWidthMismatch {
        /// Decoder hidden width.
        decoder: usize,
        /// Projector output width.
        projector: usize,
    },
    /// Projector and decoder disagree on modality token identities.
    #[error("projector modality token policy does not match the decoder")]
    ProjectorModalityMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AffineQuantization;

    fn component(name: &str, role: ArtifactRole, required: bool) -> ArtifactComponentSchema {
        ArtifactComponentSchema {
            component: ComponentId::new(name).unwrap(),
            role,
            required,
            architecture: "fixture".into(),
        }
    }

    #[test]
    fn composite_schema_preserves_required_siblings_and_rejects_collisions() {
        let schema = CompositeArtifactSchema::new(
            component("decoder", ArtifactRole::Decoder, true),
            [
                component("projector", ArtifactRole::Projector, false),
                component("assistant", ArtifactRole::Assistant, true),
            ],
        )
        .unwrap();
        assert_eq!(
            schema
                .required_components()
                .map(ComponentId::as_str)
                .collect::<Vec<_>>(),
            ["decoder", "assistant"]
        );
        assert!(CompositeArtifactSchema::new(
            component("decoder", ArtifactRole::Decoder, true),
            [component("decoder", ArtifactRole::Projector, false)],
        )
        .is_err());
    }

    #[test]
    fn component_catalog_detects_cross_artifact_parameter_collisions() {
        let decoder = ComponentId::new("decoder").unwrap();
        let projector = ComponentId::new("projector").unwrap();
        let catalog = ComponentParameterCatalog::new([
            (decoder.clone(), vec!["embed.weight".into()]),
            (projector.clone(), vec!["project.weight".into()]),
        ])
        .unwrap();
        assert_eq!(catalog.owner("project.weight"), Some(&projector));
        assert!(ComponentParameterCatalog::new([
            (decoder, vec!["shared.weight".into()]),
            (projector, vec!["shared.weight".into()]),
        ])
        .is_err());
    }

    #[test]
    fn convolution_and_quantization_schemas_are_layout_explicit() {
        assert_eq!(
            ConvolutionKernelLayout::FlattenedThreeDimensional
                .flattened_shape(&[8, 3, 2, 4, 4])
                .unwrap(),
            [8, 96]
        );
        assert!(ConvolutionKernelLayout::TwoDimensional
            .flattened_shape(&[8, 3, 3])
            .is_err());
        let affine = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
        let schema = ComponentQuantizationSchema::new(
            Some(affine),
            BTreeMap::from([("dense.weight".into(), WeightQuantization::MxFp4)]),
        )
        .unwrap();
        assert_eq!(
            schema.quantization("dense.weight"),
            Some(WeightQuantization::MxFp4)
        );
        assert_eq!(schema.quantization("other.weight"), Some(affine));
    }

    #[test]
    fn projector_compatibility_fails_closed_before_loading() {
        let compatible = ProjectorCompatibility {
            decoder_architecture: "fixture".into(),
            projector_architecture: "fixture".into(),
            decoder_hidden_width: 16,
            projector_output_width: 16,
            decoder_modality_tokens: BTreeSet::from([7, 8]),
            projector_modality_tokens: BTreeSet::from([7, 8]),
        };
        compatible.validate().unwrap();
        let mut mismatch = compatible;
        mismatch.projector_modality_tokens.remove(&8);
        assert_eq!(
            mismatch.validate().unwrap_err(),
            CompositeArtifactError::ProjectorModalityMismatch
        );
    }
}
