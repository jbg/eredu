//! The same deterministic primitives for native and metadata execution.
use super::*;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceOperationKind, WorkspaceTensor,
};
use eredu_nn::{Index, Tensor};
use eredu_runtime::speculative::numerical::SpeculativeProbabilityBackend;

pub(in crate::composition::mlx::speculative::sampling) struct Native;
fn invalid(message: std::fmt::Arguments<'_>) -> Exception {
    match safemlx::OriginalScopeObserver::try_current() {
        Ok(Some(scope)) => scope.invalid_input_error(),
        Err(cause) => cause,
        Ok(None) => Exception::custom(message.to_string()),
    }
}
impl SpeculativeProbabilityBackend for Native {
    type Value = Array;
    type Context = Stream;
    type Error = Exception;
    fn f32(value: &Array, context: &Stream) -> Result<Array, Exception> {
        value.as_type::<f32>(context)
    }
    fn softmax(value: &Array, context: &Stream) -> Result<Array, Exception> {
        softmax_axis(value, -1, true, context)
    }
    fn select(value: &Array, token: u32, context: &Stream) -> Result<Array, Exception> {
        let vocabulary = value.dim(-1);
        if vocabulary <= 0 || u64::from(token) >= vocabulary as u64 {
            return Err(invalid(format_args!(
                "sampled token {token} exceeds vocabulary size {vocabulary}"
            )));
        }
        let token = i32::try_from(token)
            .map_err(|_| invalid(format_args!("sampled token exceeds the index domain")))?;
        match value.ndim() {
            2 => value.try_index_device((0, token), context),
            3 => value.try_index_device((0, 0, token), context),
            ndim => Err(invalid(format_args!(
                "speculative distribution must be rank 2 or 3, got rank {ndim}"
            ))),
        }
    }
    fn subtract(left: &Array, right: &Array, context: &Stream) -> Result<Array, Exception> {
        left.subtract(right, context)
    }
    fn positive(value: &Array, context: &Stream) -> Result<Array, Exception> {
        maximum(value, Array::try_from_f32(0.0)?, context)
    }
    fn total(value: &Array, context: &Stream) -> Result<Array, Exception> {
        value.sum(None, context)
    }
    fn logarithm(value: &Array, context: &Stream) -> Result<Array, Exception> {
        value.log(context)
    }
}
pub(super) struct Metadata;
fn one(
    context: &WorkspaceContext,
    kind: WorkspaceOperationKind,
    inputs: &[&WorkspaceTensor],
    shape: &[i32],
) -> Result<WorkspaceTensor, eredu_nn::Error> {
    let mut outputs = context.metadata_vec(1)?;
    outputs.push(context.layout(shape, WorkspaceDtype::Float32)?);
    Ok(context.execute(kind, inputs, outputs)?.remove(0))
}
impl SpeculativeProbabilityBackend for Metadata {
    type Value = WorkspaceTensor;
    type Context = WorkspaceContext;
    type Error = eredu_nn::Error;
    fn f32(
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Self::Error> {
        one(
            context,
            WorkspaceOperationKind::Elementwise("cast_f32"),
            &[value],
            value.shape(),
        )
    }
    fn softmax(
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Self::Error> {
        value.softmax_axis(-1, true, context)
    }
    fn select(
        value: &WorkspaceTensor,
        token: u32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Self::Error> {
        let token = i32::try_from(token).map_err(|_| {
            context.metadata_error(format_args!(
                "speculative probability index exceeds native domain"
            ))
        })?;
        match value.shape().len() {
            2 => value.index(&[Index::At(0), Index::At(token)], context),
            3 => value.index(&[Index::At(0), Index::At(0), Index::At(token)], context),
            _ => Err(context.metadata_error(format_args!(
                "speculative probability source rank is invalid"
            ))),
        }
    }
    fn subtract(
        left: &WorkspaceTensor,
        right: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Self::Error> {
        left.subtract(right, context)
    }
    fn positive(
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Self::Error> {
        value.maximum_scalar(0.0, context)
    }
    fn total(
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Self::Error> {
        one(
            context,
            WorkspaceOperationKind::Reduction("sum_all", 0, false),
            &[value],
            &[],
        )
    }
    fn logarithm(
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Self::Error> {
        one(
            context,
            WorkspaceOperationKind::Elementwise("log"),
            &[value],
            value.shape(),
        )
    }
}
