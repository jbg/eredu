//! Retained codec parameter scopes and their component execution projections.
use crate::{
    ear_specs::{DecoderSpec, EncoderSpec},
    prvq_specs::QuantizerSpec,
    Error,
};
use eredu_runtime::{ComponentDefinition, ComponentPreparationDeclaration};

/// Immutable typed codec parameters paired with normalized invocation policy.
#[derive(Debug, Clone)]
pub struct CodecDeclaration<S, C> {
    config: C,
    prefix: String,
    spec: S,
    definition: ComponentDefinition,
}
impl<S, C> CodecDeclaration<S, C> {
    /// Normalized component policy retained with these parameters.
    pub fn config(&self) -> &C {
        &self.config
    }
    /// Canonical parameter namespace.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
    /// Authoritative typed neural construction slots.
    pub fn spec(&self) -> &S {
        &self.spec
    }
    /// Parameter, invocation, state and placement projection.
    pub fn definition(&self) -> &ComponentDefinition {
        &self.definition
    }
    pub(crate) fn into_parts(self) -> (C, String, S, ComponentDefinition) {
        (self.config, self.prefix, self.spec, self.definition)
    }
}
impl<S, C> ComponentPreparationDeclaration for CodecDeclaration<S, C> {
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}
/// EAR spectral encoder scopes and their execution contract.
pub type EncoderDeclaration = CodecDeclaration<EncoderSpec, crate::ear_encoder::EncoderConfig>;
/// EAR neural decoder scopes and their execution contract.
pub type DecoderDeclaration = CodecDeclaration<DecoderSpec, crate::ear_decoder::DecoderConfig>;
/// Independently leased PRVQ codebook scopes and their execution contract.
pub type QuantizerDeclaration = CodecDeclaration<QuantizerSpec, crate::prvq::Geometry>;
impl EncoderDeclaration {
    /// Projects exact rank-local dimensions from this declaration and retained placement.
    pub fn local_dimensions(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<crate::ear_encoder_layered::EncoderLocalDimensions, Error> {
        crate::ear_encoder_layered::declared_local_dimensions(&self.spec, layout)
    }

    /// Builds and validates source-free encoder declarations before allocation.
    pub fn new(config: crate::ear_encoder::EncoderConfig, prefix: &str) -> Result<Self, Error> {
        if prefix.is_empty() {
            return Err(Error::InvalidShape(
                "empty EAR encoder parameter prefix".into(),
            ));
        }
        let spec = EncoderSpec::new(&config, prefix)?;
        let definition = crate::ear_encoder_layered::declared_component_definition(prefix, &spec)?;
        Ok(Self {
            config,
            prefix: prefix.into(),
            spec,
            definition,
        })
    }
}
impl DecoderDeclaration {
    /// Projects exact rank-local dimensions from this declaration and retained placement.
    pub fn local_dimensions(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<crate::ear_decoder_layered::DecoderLocalDimensions, Error> {
        crate::ear_decoder_layered::declared_local_dimensions(&self.spec, layout)
    }

    /// Builds and validates source-free decoder declarations before allocation.
    pub fn new(config: crate::ear_decoder::DecoderConfig, prefix: &str) -> Result<Self, Error> {
        if prefix.is_empty() {
            return Err(Error::InvalidShape(
                "empty EAR decoder parameter prefix".into(),
            ));
        }
        let spec = DecoderSpec::new(&config, prefix)?;
        let definition =
            crate::ear_decoder_layered::declared_component_definition(&config, prefix, &spec)?;
        Ok(Self {
            config,
            prefix: prefix.into(),
            spec,
            definition,
        })
    }
}
impl QuantizerDeclaration {
    /// Projects exact rank-local dimensions from this declaration and retained placement.
    pub fn local_dimensions(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<crate::prvq_layered::QuantizerLocalDimensions, Error> {
        crate::prvq_layered::declared_local_dimensions(&self.spec, layout)
    }

    /// Builds and validates source-free quantizer declarations before allocation.
    pub fn new(geometry: crate::prvq::Geometry, prefix: &str) -> Result<Self, Error> {
        let spec = QuantizerSpec::new(geometry, prefix)?;
        let definition = crate::prvq_layered::declared_component_definition(&spec)?;
        Ok(Self {
            config: geometry,
            prefix: prefix.into(),
            spec,
            definition,
        })
    }
}
