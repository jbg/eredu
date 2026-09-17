use super::*;
use eredu_nn::{
    Tensor,
    workspace::{WorkspaceDtype as D, WorkspaceOperationKind as K},
};
pub(super) struct Trace<'a> {
    pub(super) context: &'a WorkspaceContext,
    pub(super) retained: &'a mut Vec<WorkspaceTensor>,
}
impl Trace<'_> {
    fn retained(&mut self, value: WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        self.context.reserve_metadata_vec(self.retained, 1)?;
        self.retained.push(value.clone());
        Ok(value)
    }
    fn operation(
        &mut self,
        kind: K,
        inputs: &[&WorkspaceTensor],
        shape: &[i32],
        dtype: D,
    ) -> Result<WorkspaceTensor, Error> {
        let mut outputs = self.context.metadata_vec(1)?;
        outputs.push(self.context.layout(shape, dtype)?);
        let value = self.context.execute(kind, inputs, outputs)?.remove(0);
        self.retained(value)
    }
}
impl Kernel for Trace<'_> {
    type Value = WorkspaceTensor;
    fn interval(
        &mut self,
        source: &WorkspaceTensor,
        start: i32,
        end: i32,
    ) -> Result<WorkspaceTensor, Error> {
        self.retained(source.static_slice(&[start], &[end], &[1], self.context)?)
    }
    fn unary(&mut self, kind: Unary, source: &WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        let (kind, shape, dtype) = match kind {
            Unary::CastF32 => (
                K::Elementwise("capture_cast_f32"),
                source.shape(),
                D::Float32,
            ),
            Unary::Finite => (K::Elementwise("is_finite"), source.shape(), D::Bool),
            Unary::Nan => (K::Elementwise("is_nan"), source.shape(), D::Bool),
            Unary::PositiveInfinity => (
                K::Elementwise("is_positive_infinity"),
                source.shape(),
                D::Bool,
            ),
            Unary::NegativeInfinity => (
                K::Elementwise("is_negative_infinity"),
                source.shape(),
                D::Bool,
            ),
            Unary::MaskU32 => (K::Elementwise("bool_to_u32"), source.shape(), D::Uint32),
            Unary::Sum => (
                K::Reduction("sum", 0, false),
                &[][..],
                source.layout().dtype(),
            ),
            Unary::Minimum => (
                K::Reduction("min", 0, false),
                &[][..],
                source.layout().dtype(),
            ),
            Unary::Maximum => (
                K::Reduction("max", 0, false),
                &[][..],
                source.layout().dtype(),
            ),
        };
        self.operation(kind, &[source], shape, dtype)
    }
    fn scalar(&mut self, _: f32) -> Result<WorkspaceTensor, Error> {
        self.operation(K::Elementwise("scalar_f32"), &[], &[], D::Float32)
    }
    fn select(
        &mut self,
        mask: &WorkspaceTensor,
        source: &WorkspaceTensor,
        other: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        self.operation(
            K::Elementwise("where"),
            &[mask, source, other],
            source.shape(),
            D::Float32,
        )
    }
    fn binary(
        &mut self,
        kind: Binary,
        left: &WorkspaceTensor,
        right: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        self.operation(
            K::Elementwise(match kind {
                Binary::Divide => "divide",
                Binary::Multiply => "multiply",
            }),
            &[left, right],
            left.shape(),
            D::Float32,
        )
    }
    fn read_u32(&mut self, _: &WorkspaceTensor, representative: u32) -> Result<u32, Error> {
        Ok(representative)
    }
    fn read_f32(&mut self, _: &WorkspaceTensor, kind: ScalarRead) -> Result<f64, Error> {
        Ok(kind.representative())
    }
}
