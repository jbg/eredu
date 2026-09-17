use super::*;
use eredu_core::InputPayloadKind;
use eredu_runtime::input::host::{HostInputPartView, HostTensorValues, HostTensorView};

pub(super) struct ProcessorPart<'a>(pub(super) &'a PreparedInputPart<HostTensor>);
fn view(t: &HostTensor) -> HostTensorView<'_> {
    match t {
        HostTensor::U32 { values, shape } => HostTensorView {
            shape,
            values: HostTensorValues::U32(values),
        },
        HostTensor::I32 { values, shape } => HostTensorView {
            shape,
            values: HostTensorValues::I32(values),
        },
        HostTensor::F32 { values, shape } => HostTensorView {
            shape,
            values: HostTensorValues::F32(values),
        },
        HostTensor::Bool { values, shape } => HostTensorView {
            shape,
            values: HostTensorValues::Bool(values),
        },
    }
}
impl HostInputPartView for ProcessorPart<'_> {
    fn modality(&self) -> InputModality {
        self.0.modality()
    }
    fn kind(&self) -> InputPayloadKind {
        self.0.payload().kind()
    }
    fn payload(&self) -> HostTensorView<'_> {
        view(self.0.payload().value())
    }
    fn metadata(&self) -> impl ExactSizeIterator<Item = (InputMetadataKey, HostTensorView<'_>)> {
        self.0.metadata().iter().map(|(key, t)| (*key, view(t)))
    }
    fn extents(&self) -> &[InputExtent] {
        self.0.extents()
    }
}
/// Ordinary shared lowering for actual ordered prepared host parts. This same
/// path serves existing processor outputs and closed original host-source views.
/// It creates native tensors/parts/identities and carries no memory grant: the
/// caller must retain the appropriate operation and partial-native owners.
pub fn lower_prepared_host_input<M, E, P, I>(
    parts: I,
    mechanisms: &mut M,
) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorMechanisms,
    E: std::fmt::Display,
    P: HostInputPartView,
    I: ExactSizeIterator<Item = P> + Clone,
{
    if parts.clone().any(|p| {
        !matches!(
            p.kind(),
            InputPayloadKind::TokenIds | InputPayloadKind::Tensor | InputPayloadKind::Embeddings
        )
    }) {
        return Err(ProcessorExecutionError::Plan(
            "host processor emitted an unsupported payload kind".into(),
        ));
    }
    let mut output = Vec::with_capacity(parts.len());
    for part in parts {
        let value = lower_tensor(part.payload(), mechanisms)?;
        let payload = match part.kind() {
            InputPayloadKind::TokenIds => PreparedInputPayload::TokenIds(value),
            InputPayloadKind::Tensor => PreparedInputPayload::Tensor(value),
            InputPayloadKind::Embeddings => PreparedInputPayload::Embeddings(value),
            _ => {
                return Err(ProcessorExecutionError::Plan(
                    "host processor emitted an unsupported payload kind".into(),
                ))
            }
        };
        let metadata = part
            .metadata()
            .map(|(key, value)| lower_tensor(value, mechanisms).map(|value| (key, value)))
            .collect::<Result<Vec<_>, _>>()?;
        output.push(
            PreparedInputPart::new_with_extents(
                part.modality(),
                payload,
                metadata,
                part.extents().iter().copied(),
            )
            .map_err(ProcessorExecutionError::Prepared)?,
        );
    }
    PreparedModelInput::new(output, |tensor| mechanisms.identity(tensor))
        .map_err(ProcessorExecutionError::Prepared)
}
fn lower_tensor<M, E>(
    tensor: HostTensorView<'_>,
    mechanisms: &mut M,
) -> Result<M::Tensor, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorMechanisms,
    E: std::fmt::Display,
{
    match tensor.values {
        HostTensorValues::U32(values) => mechanisms
            .tensor_u32(values, tensor.shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensorValues::I32(values) => mechanisms
            .tensor_i32(values, tensor.shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensorValues::F32(values) => mechanisms
            .tensor_f32(values, tensor.shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensorValues::Bool(values) => mechanisms
            .tensor_bool(values, tensor.shape)
            .map_err(optional_mechanism_error),
    }
}

#[cfg(test)]
mod tests;

/// Exact output of lowering a genuine original source with the shared mechanism.
/// ```compile_fail
/// use eredu_architectures::processor_execution::OriginalHostLowering;
/// fn cannot_relabel<T>(input:eredu_runtime::PreparedModelInput<T>,source:eredu_runtime::working_memory::OriginalPreparedHostInput) {
///     let _=OriginalHostLowering { prepared:input,source };
/// }
/// ```
/// No raw constructor, mutable tensor/part accessor, or source relabelling exists.
/// Cloning an ordinary lowering preserves its allocating behavior; originally constructed owners share their existing source controls.
#[derive(Clone)]
pub struct OriginalHostLowering<T> {
    prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
    source: eredu_runtime::working_memory::OriginalPreparedHostInput,
}
impl<T> OriginalHostLowering<T> {
    /// Accepts only an already completed source-owned prepared view. Ordinary
    /// public views have no source certificate and are returned unchanged.
    pub fn from_original_owner(
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
    ) -> Result<Self, eredu_runtime::input::PreparedModelInputOwner<T>> {
        let Some(source) = prepared.original_source().cloned() else {
            return Err(prepared);
        };
        Ok(Self { prepared, source })
    }
    pub fn prepared(&self) -> &PreparedModelInput<T> {
        &self.prepared
    }
    pub fn source(&self) -> &eredu_runtime::working_memory::OriginalPreparedHostInput {
        &self.source
    }
    pub fn into_prepared(self) -> eredu_runtime::input::PreparedModelInputOwner<T> {
        self.prepared
    }
}
/// Shares the ordinary lowerer and retains actual source identity. A backend
/// must still establish completion under its real ordinary owner before exposing
/// the native result; this wrapper grants neither completion nor managed work.
pub fn lower_original_prepared_host_input<M, E>(
    source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
    mechanisms: &mut M,
) -> Result<OriginalHostLowering<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorMechanisms,
    E: std::fmt::Display,
{
    let prepared = lower_prepared_host_input(source.parts(), mechanisms)?;
    Ok(OriginalHostLowering {
        prepared: prepared.into(),
        source: source.clone(),
    })
}
