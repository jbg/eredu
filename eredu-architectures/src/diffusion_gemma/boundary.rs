//! Shared-stack invocation semantics transported across pipeline cuts.
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    BoundaryTensorDimension, BoundaryTensorDtype, BoundaryTensorSpec,
};

/// Invocation phase and authoritative rotary frontier, encoded as two I32 values.
#[derive(Debug, Clone)]
pub struct TextBoundary<T> {
    /// `[phase, position]`, where phase zero appends context and one refines it.
    pub invocation: T,
}

/// The same wire contract applies to context and canvas invocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextBoundarySchema {
    hidden: i32,
}
impl TextBoundarySchema {
    /// Declares the shared residual width independently of local head ownership.
    pub fn new(hidden: i32) -> Self {
        Self { hidden }
    }
}
impl ArchitectureBoundary for TextBoundarySchema {
    type Boundary<T> = TextBoundary<T>;
    const IDENTITY: &'static str = "diffusion_gemma.shared_text";
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::primary_activation(self.hidden)
    }
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        vec![BoundaryTensorSpec::new(
            "invocation",
            [BoundaryTensorDimension::Fixed(2)],
            BoundaryTensorDtype::Int32,
        )]
    }
    fn encode<T>(
        &self,
        boundary: TextBoundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(vec![ArchitectureBoundaryValue::new(
            "invocation",
            boundary.invocation,
        )?])
    }
    fn decode<T>(&self, tensors: Vec<T>) -> Result<TextBoundary<T>, ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(TextBoundary {
            invocation: tensors
                .into_iter()
                .next()
                .expect("validated invocation cardinality"),
        })
    }
}
