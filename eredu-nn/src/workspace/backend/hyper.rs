use super::*;
use crate::{
    HyperConnectionOperator, HyperConnectionSpec, HyperConnectionState, HyperHeadOperator,
    HyperHeadSpec, HyperNeuralBackend, TensorValueObserver,
};

/// Metadata for one multi-stream residual cycle and its learned mixing policy.
#[derive(Clone, Debug, crate::Parameterized)]
#[parameterized(tensor = "WorkspaceTensor")]
pub struct WorkspaceHyperConnection {
    parameters: Vec<Parameter<WorkspaceTensor>>,
    #[parameter(skip, metadata)]
    spec: HyperConnectionSpec,
}
/// Metadata for the final learned stream collapse, including coefficient hooks.
#[derive(Clone, Debug, crate::Parameterized)]
#[parameterized(tensor = "WorkspaceTensor")]
pub struct WorkspaceHyperHead {
    parameters: Vec<Parameter<WorkspaceTensor>>,
    #[parameter(skip, metadata)]
    spec: HyperHeadSpec,
}
fn residual_shape(
    residual: &WorkspaceTensor,
    streams: i32,
    hidden: i32,
    context: &WorkspaceContext,
) -> Result<([i32; 3], [i32; 3], [i32; 4]), Error> {
    let shape = residual.shape();
    if shape.len() != 4 || shape[2..] != [streams, hidden] {
        return Err(context.metadata_error(format_args!(
            "workspace hyper residual differs from declared stream geometry"
        )));
    }
    Ok((
        [shape[0], shape[1], hidden],
        [shape[0], shape[1], streams],
        [shape[0], shape[1], streams, streams],
    ))
}
fn parameters(
    function: &ParameterSpec,
    base: &ParameterSpec,
    scale: &ParameterSpec,
    rows: i32,
    columns: i32,
    scales: i32,
    context: &WorkspaceContext,
) -> Result<Vec<Parameter<WorkspaceTensor>>, Error> {
    let shapes: [(&ParameterSpec, &[i32]); 3] = [
        (function, &[rows, columns]),
        (base, &[rows]),
        (scale, &[scales]),
    ];
    let mut parameters = context.metadata_vec(shapes.len())?;
    for (spec, shape) in shapes {
        parameters.push(parameter(
            context.clone_metadata(spec)?,
            shape,
            WorkspaceDtype::Float32,
            context,
        )?);
    }
    Ok(parameters)
}
fn width(streams: i32, hidden: i32, context: &WorkspaceContext) -> Result<i32, Error> {
    streams.checked_mul(hidden).ok_or_else(|| {
        context.metadata_error(format_args!("workspace hyper projection width overflow"))
    })
}
impl HyperNeuralBackend for WorkspaceBackend {
    type HyperConnection = WorkspaceHyperConnection;
    type HyperHead = WorkspaceHyperHead;
    fn hyper_connection(
        spec: HyperConnectionSpec,
        context: &WorkspaceContext,
    ) -> Result<Self::HyperConnection, Error> {
        spec.validate()?;
        let columns = width(spec.streams, spec.hidden_size, context)?;
        let rows = spec
            .streams
            .checked_add(2)
            .and_then(|n| n.checked_mul(spec.streams))
            .ok_or_else(|| {
                context.metadata_error(format_args!("workspace hyper mixing width overflow"))
            })?;
        let parameters = parameters(
            &spec.function,
            &spec.base,
            &spec.scale,
            rows,
            columns,
            3,
            context,
        )?;
        Ok(WorkspaceHyperConnection { parameters, spec })
    }
    fn hyper_head(
        spec: HyperHeadSpec,
        context: &WorkspaceContext,
    ) -> Result<Self::HyperHead, Error> {
        spec.validate()?;
        let columns = width(spec.streams, spec.hidden_size, context)?;
        let parameters = parameters(
            &spec.function,
            &spec.base,
            &spec.scale,
            spec.streams,
            columns,
            1,
            context,
        )?;
        Ok(WorkspaceHyperHead { parameters, spec })
    }
}
impl HyperConnectionOperator<WorkspaceTensor> for WorkspaceHyperConnection {
    fn collapse(
        &mut self,
        residual: &WorkspaceTensor,
        norm_epsilon: f32,
        context: &WorkspaceContext,
    ) -> Result<HyperConnectionState<WorkspaceTensor>, Error> {
        if !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
            return Err(context.metadata_error(format_args!(
                "workspace hyper RMS epsilon must be finite and positive"
            )));
        }
        let (collapsed, coefficients, combination) =
            residual_shape(residual, self.spec.streams, self.spec.hidden_size, context)?;
        let mut inputs = context.metadata_vec(
            self.parameters
                .len()
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        inputs.push(residual);
        inputs.extend(self.parameters.iter().map(Parameter::as_ref));
        let shapes: [&[i32]; 4] = [&collapsed, &coefficients, &coefficients, &combination];
        let mut layouts = context.metadata_vec(shapes.len())?;
        for shape in shapes {
            layouts.push(context.layout(shape, WorkspaceDtype::Float32)?);
        }
        let mut outputs = context
            .execute(
                WorkspaceOperationKind::HyperCollapse(
                    context.box_metadata(context.clone_metadata(&self.spec)?)?,
                    norm_epsilon,
                ),
                &inputs,
                layouts,
            )?
            .into_iter();
        Ok(HyperConnectionState {
            collapsed: outputs.next().unwrap(),
            pre: outputs.next().unwrap(),
            post: outputs.next().unwrap(),
            combination: outputs.next().unwrap(),
        })
    }
    fn expand(
        &mut self,
        sublayer: &WorkspaceTensor,
        residual: &WorkspaceTensor,
        state: &HyperConnectionState<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        let (collapsed, coefficients, combination) =
            residual_shape(residual, self.spec.streams, self.spec.hidden_size, context)?;
        if sublayer.shape() != collapsed
            || state.collapsed.shape() != collapsed
            || state.pre.shape() != coefficients
            || state.post.shape() != coefficients
            || state.combination.shape() != combination
        {
            return Err(context.metadata_error(format_args!(
                "workspace hyper state differs from residual geometry"
            )));
        }
        context.validate_values([&state.collapsed, &state.pre])?;
        WorkspaceTensor::operation(
            WorkspaceOperationKind::HyperExpand,
            &[sublayer, residual, &state.post, &state.combination],
            residual.shape(),
            WorkspaceDtype::Float32,
            context,
        )
    }
}
impl HyperHeadOperator<WorkspaceTensor> for WorkspaceHyperHead {
    fn forward(
        &mut self,
        residual: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.forward_with_coefficients_observer(residual, context, None)
    }
    fn forward_with_coefficients_observer(
        &mut self,
        residual: &WorkspaceTensor,
        context: &WorkspaceContext,
        observer: Option<&mut dyn TensorValueObserver<WorkspaceTensor>>,
    ) -> Result<WorkspaceTensor, Error> {
        let (collapsed, coefficients, _) =
            residual_shape(residual, self.spec.streams, self.spec.hidden_size, context)?;
        let mut inputs = context.metadata_vec(
            self.parameters
                .len()
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        inputs.push(residual);
        inputs.extend(self.parameters.iter().map(Parameter::as_ref));
        let coefficients = WorkspaceTensor::operation(
            WorkspaceOperationKind::HyperHeadCoefficients(
                context.box_metadata(context.clone_metadata(&self.spec)?)?,
            ),
            &inputs,
            &coefficients,
            WorkspaceDtype::Float32,
            context,
        )?;
        if let Some(observer) = observer {
            observer.observe(&coefficients)?;
        }
        WorkspaceTensor::operation(
            WorkspaceOperationKind::HyperHeadSum,
            &[residual, &coefficients],
            &collapsed,
            WorkspaceDtype::Float32,
            context,
        )
    }
}
