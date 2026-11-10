//! Fixed-mean residual codebook owners shared by construction and cold projections.
use crate::{
    prvq::{invalid, Geometry},
    Error,
};
use eredu_nn::{
    DeclareLogicalParameters, LogicalParameterDeclaration, LogicalParameterRole, ParameterSpec,
};
fn validate(spec: &impl DeclareLogicalParameters) -> Result<(), Error> {
    eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
        .map_err(|error| invalid(error.to_string()))
}
/// One frozen mean table and its independently retained scalar variance.
#[derive(Debug, Clone)]
pub struct CodebookSpec {
    pub(crate) geometry: Geometry,
    pub(crate) depth: usize,
    pub(crate) prefix: String,
    means: ParameterSpec,
    variance: ParameterSpec,
}
impl CodebookSpec {
    /// Declares one canonical residual depth without allocating learned tensors.
    pub fn new(geometry: Geometry, prefix: &str, depth: usize) -> Result<Self, Error> {
        geometry.validate()?;
        if prefix.is_empty() || depth >= geometry.depth {
            return Err(invalid("invalid codebook parameter prefix or depth"));
        }
        let parameter = |name: String| {
            let mut spec = ParameterSpec::trainable(name).map_err(|e| invalid(e.to_string()))?;
            spec.trainable = false;
            spec.group = Some(format!("{prefix}.codebook.{depth}"));
            Ok::<_, Error>(spec)
        };
        let spec = Self {
            geometry,
            depth,
            prefix: prefix.into(),
            means: parameter(format!("{prefix}.mus_list.{depth}"))?,
            variance: parameter(format!("{prefix}._variance_list.{depth}.variance"))?,
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Selects table channels while retaining the complete variance scalar.
    pub fn local(mut self, channels: i32) -> Result<Self, Error> {
        if channels <= 0 || channels > self.geometry.channels {
            return Err(invalid("invalid residual codebook channel slice"));
        }
        self.geometry.channels = channels;
        validate(&self)?;
        Ok(self)
    }
    /// Declared mean-table identity and ownership metadata.
    pub fn means(&self) -> &ParameterSpec {
        &self.means
    }
    /// Declared retained-variance identity and ownership metadata.
    pub fn variance(&self) -> &ParameterSpec {
        &self.variance
    }
    /// Exact selected geometry of this codebook.
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }
    /// Ordered global residual depth.
    pub fn depth(&self) -> usize {
        self.depth
    }
    /// Logical variance geometry; singleton source axes belong to storage extensions.
    pub fn variance_shape(&self) -> [i32; 0] {
        []
    }
    /// Mean-vector geometry, independent of physical source encoding.
    pub fn mean_shape(&self) -> [i32; 2] {
        [self.geometry.cardinality, self.geometry.channels]
    }
}
impl DeclareLogicalParameters for CodebookSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        Ok(vec![
            LogicalParameterDeclaration::new(
                &self.means,
                &self.mean_shape(),
                LogicalParameterRole::Codebook,
            )?,
            LogicalParameterDeclaration::new(
                &self.variance,
                &self.variance_shape(),
                LogicalParameterRole::CodebookVariance,
            )?,
        ])
    }
}
/// Complete ownership of an ordered sequence of independently leased codebooks.
#[derive(Debug, Clone)]
pub struct QuantizerSpec {
    pub(crate) geometry: Geometry,
    pub(crate) prefix: String,
    pub(crate) codebooks: Vec<CodebookSpec>,
}
impl QuantizerSpec {
    /// Validates the complete parameter topology before any codebook allocation.
    pub fn new(geometry: Geometry, prefix: &str) -> Result<Self, Error> {
        geometry.validate()?;
        if prefix.is_empty() {
            return Err(invalid("empty residual quantizer parameter prefix"));
        }
        let spec = Self {
            geometry,
            prefix: prefix.into(),
            codebooks: (0..geometry.depth)
                .map(|depth| CodebookSpec::new(geometry, prefix, depth))
                .collect::<Result<_, _>>()?,
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Ordered global owners; execution invocations retain separate phase identities.
    pub fn codebooks(&self) -> &[CodebookSpec] {
        &self.codebooks
    }
    /// Shared codec geometry.
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }
    /// Projects canonical parameter groups and residency ownership from these slots.
    pub fn parameter_description(
        &self,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Error> {
        crate::prvq_layered::declared_parameters(self)
    }
}
impl DeclareLogicalParameters for QuantizerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        self.codebooks[..].logical_parameters()
    }
}
