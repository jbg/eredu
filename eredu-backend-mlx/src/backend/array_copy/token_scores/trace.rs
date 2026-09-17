use super::*;
use eredu_nn::{
    Tensor,
    workspace::{
        WorkspaceDtype as D, WorkspaceOperationKind as K, WorkspaceSamplingOperation as S,
    },
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
    fn flatten(
        &mut self,
        source: &WorkspaceTensor,
        vocabulary: i32,
    ) -> Result<WorkspaceTensor, Error> {
        self.retained(source.reshape(&[-1, vocabulary], self.context)?)
    }
    fn terminal(&mut self, source: &WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        let rows = source.shape()[0];
        let width = source.shape()[1];
        let selected = source.static_slice(&[rows - 1, 0], &[rows, width], &[1, 1], self.context)?;
        self.retained(selected.reshape(&[width], self.context)?)
    }
    fn unary(&mut self, kind: Unary, source: &WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        let (kind, shape, dtype) = match kind {
            Unary::CastF32 => (
                K::Elementwise("capture_cast_f32"),
                source.shape(),
                D::Float32,
            ),
            Unary::Finite => (K::Elementwise("is_finite"), source.shape(), D::Bool),
            Unary::MaskU32 => (K::Elementwise("bool_to_u32"), source.shape(), D::Uint32),
            Unary::Sum => (
                K::Reduction("sum", 0, false),
                &[][..],
                source.layout().dtype(),
            ),
            Unary::Maximum => (
                K::Reduction("max", 0, false),
                &[][..],
                source.layout().dtype(),
            ),
            Unary::Exp => (K::Elementwise("exp"), source.shape(), D::Float32),
            Unary::Argmax => (K::Sampling(S::Greedy), &[][..], D::Uint32),
        };
        self.operation(kind, &[source], shape, dtype)
    }
    fn scalar(&mut self, _: f32) -> Result<WorkspaceTensor, Error> {
        self.operation(K::Elementwise("scalar_f32"), &[], &[], D::Float32)
    }
    fn interval(
        &mut self,
        row: &WorkspaceTensor,
        start: i32,
        end: i32,
    ) -> Result<WorkspaceTensor, Error> {
        self.retained(row.static_slice(&[start], &[end], &[1], self.context)?)
    }
    fn at(&mut self, row: &WorkspaceTensor, id: u32) -> Result<WorkspaceTensor, Error> {
        let selected = row.static_slice(&[id as i32], &[id as i32 + 1], &[1], self.context)?;
        self.retained(selected.reshape(&[], self.context)?)
    }
    fn binary(
        &mut self,
        kind: Binary,
        left: &WorkspaceTensor,
        right: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        let (kind, dtype) = match kind {
            Binary::Subtract => (K::Elementwise("subtract"), D::Float32),
            Binary::Greater => (K::Elementwise("greater"), D::Bool),
        };
        self.operation(kind, &[left, right], left.shape(), dtype)
    }
    fn exclude(
        &mut self,
        row: &WorkspaceTensor,
        id: u32,
        value: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        // The actual static scalar assignment first expands its scalar to [1]
        // and uses the existing shared SliceUpdate worker. Count that real view
        // separately. The exact CPU source elides the same-dtype cast and
        // same-shape broadcast, then prices the actual scalar overwrite.
        let update = self.retained(value.reshape(&[1], self.context)?)?;
        // The actual indexed-update wrapper owns a separate row handle before
        // replacing it; this clone contributes that real Graph shell.
        let alias = row.clone();
        self.retained(alias.update_slice(&update, &[id as i32], self.context)?)
    }
    fn read_f32(&mut self, _: &WorkspaceTensor) -> Result<f64, Error> {
        Ok(0.0)
    }
    fn read_u32(&mut self, _: &WorkspaceTensor, kind: Read) -> Result<u32, Error> {
        Ok(kind.representative())
    }
}
