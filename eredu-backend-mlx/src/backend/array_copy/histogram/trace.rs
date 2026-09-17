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
            Unary::Not => (K::Elementwise("logical_not"), source.shape(), D::Bool),
            Unary::MaskU32 => (K::Elementwise("bool_to_u32"), source.shape(), D::Uint32),
            Unary::Sum => (K::Reduction("sum", 0, false), &[][..], D::Uint32),
        };
        self.operation(kind, &[source], shape, dtype)
    }
    fn scalar(&mut self, _: f32) -> Result<WorkspaceTensor, Error> {
        self.operation(K::Elementwise("scalar_f32"), &[], &[], D::Float32)
    }
    fn binary(
        &mut self,
        kind: Binary,
        left: &WorkspaceTensor,
        right: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        self.operation(
            K::Elementwise(match kind {
                Binary::Less => "less",
                Binary::Greater => "greater",
                Binary::GreaterEqual => "greater_equal",
                Binary::LessEqual => "less_equal",
                Binary::And => "logical_and",
            }),
            &[left, right],
            left.shape(),
            D::Bool,
        )
    }
    fn read(&mut self, _: &WorkspaceTensor) -> Result<u32, Error> {
        Ok(0)
    }
}
